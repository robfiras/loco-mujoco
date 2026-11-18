# Gymnasium interface

Every LocoMuJoCo environment can be used through the [Gymnasium](https://gymnasium.farama.org) API, which
makes it available to training libraries that are written against that API. Importing `loco_mujoco`
registers a single Gymnasium environment with the id `LocoMujoco`. The robot is chosen with the argument
`env_name`.

```{literalinclude} ../../examples/tutorials/04_creating_gymansium_env.py
:language: python
```

## Creating an environment

```python
import gymnasium as gym
import loco_mujoco          # registers the id "LocoMujoco"

env = gym.make("LocoMujoco", env_name="UnitreeH1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

All keyword arguments other than `render_mode` are passed on to a [task factory](task_factories.md), so
everything described in this user guide (observations, rewards, control functions, terrain and the other
modules) is configured in the same way as without Gymnasium:

| Argument | Meaning |
|---|---|
| `env_name` | Name of a registered LocoMuJoCo environment, for example `"UnitreeG1"` or `"SkeletonTorque"` |
| `render_mode` | `None` (default), `"human"` or `"rgb_array"` |
| `default_dataset_conf`, `amass_dataset_conf`, `lafan1_dataset_conf`, `custom_dataset_conf` | Dataset configurations. If at least one is given, the environment is created with {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` and loads the trajectories. Otherwise {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` is used. |
| any other keyword | Passed to the factory and from there to the environment, for example `reward_type`, `goal_type`, `control_type`, `observation_spec` or `terrain_type` |

```python
from loco_mujoco.task_factories import DefaultDatasetConf

env = gym.make("LocoMujoco", env_name="UnitreeH1",
               default_dataset_conf=DefaultDatasetConf("walk"),
               goal_type="GoalTrajMimic", reward_type="MimicReward")
```

## How the API is mapped

The wrapper is {class}`~loco_mujoco.core.wrappers.gymnasium.GymnasiumWrapper`. It translates between the two
interfaces as follows:

| Gymnasium | LocoMuJoCo |
|---|---|
| `observation_space`, `action_space` | Boxes with the shapes of `env.info.observation_space` and `env.info.action_space` and the dtype `float64`. Each box has one scalar lower and upper bound, the smallest and the largest bound of the original space. With the built-in control functions the action space is $[-1, 1]$. |
| `reset(seed=...)` | Calls `reset` and returns the observation and an empty info dictionary. The seed is used for NumPy's global random state, which the CPU environments draw from. Without a seed a random one is chosen. |
| `terminated` | The `absorbing` flag, see [Absorbing and done](initial_terminal_states.md#absorbing-and-done) |
| `truncated` | The `done` flag. It is true when the horizon is reached or the reference clip ends, and it is also true whenever `terminated` is true. |
| `render()` | With `render_mode="human"` the viewer window is drawn. With `"rgb_array"` the scene is rendered off-screen and the frame is returned as an array of shape (width, height, 3). |
| `close()` | Calls `stop` |
| `unwrapped` | The LocoMuJoCo environment itself |

Methods that Gymnasium does not know are reached through `env.unwrapped`, for example
`env.unwrapped.play_trajectory(n_episodes=3)` or `env.unwrapped.obs_container`.

## Limitations

- **One environment on the CPU.** The wrapper drives the CPU step functions `reset` and `step`. Parallel
  simulation with MJX is not available through Gymnasium. Use the native MJX interface and the wrappers
  below for that.
- **The viewer is selected by `render_mode`.** The argument `headless` is not accepted. With
  `render_mode=None` and `"rgb_array"` the environment is created headless.
- **Rendering in `rgb_array` mode records.** The frames are also written to a video, which is saved when
  the environment is closed, see [Recording videos](rendering.md#recording-videos).
- **No time limit wrapper is added.** The episode length is the `horizon` of the environment, 1000 steps
  by default, and it is reported through `truncated`.

## Wrappers for MJX environments

The JAX algorithms do not use Gymnasium. They use a small set of functional wrappers from
{mod}`loco_mujoco.core.wrappers.mjx` around the MJX step functions. All of them share one interface, in
which the state is passed in and returned explicitly:

```python
obs, state = env.reset(rng_key, traj)
obs, reward, absorbing, done, info, state = env.step(state, action, traj)
```

The argument `traj` is the reference trajectory and can be left out for environments without one. Wrapping
an MJX environment in any of these wrappers first converts `mjx_reset` and `mjx_step` to this interface.
When an episode ends, the returned observation is the last one of the finished episode, while `state`
already belongs to the new episode.

| Wrapper | Purpose |
|---|---|
| {class}`~loco_mujoco.core.wrappers.mjx.VecEnv` | Maps `reset` and `step` over a batch of environments with `jax.vmap`. Keys, states and actions get a leading batch dimension, the trajectory is shared. |
| {class}`~loco_mujoco.core.wrappers.mjx.LogWrapper` | Accumulates the return and the length of every episode in `state.metrics`. The fields `returned_episode_returns` and `returned_episode_lengths` hold the values of the last finished episode. |
| {class}`~loco_mujoco.core.wrappers.mjx.NStepWrapper` | Adds a history of past observations to the observation and updates `info.observation_space` accordingly |

```python
import jax
import jax.numpy as jnp
from loco_mujoco import ImitationFactory
from loco_mujoco.core.wrappers import LogWrapper, VecEnv

env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="walk"),
                                  goal_type="GoalTrajMimic", reward_type="MimicReward")
wrapped = VecEnv(LogWrapper(env))

n_envs = 4
keys = jax.random.split(jax.random.key(0), n_envs)
obs, state = wrapped.reset(keys, traj)                       # obs has the shape (n_envs, obs_dim)
action = jnp.zeros((n_envs, env.info.action_space.shape[0]))
obs, reward, absorbing, done, info, state = wrapped.step(state, action, traj)
```

The wrappers are nested from the inside out as `VecEnv(LogWrapper(NStepWrapper(env, ...)))`, which is the
order the training algorithms use. A wrapper forwards unknown attributes to the environment it wraps, so
`wrapped.info` or `wrapped.obs_container` work, and the state of every wrapper forwards unknown fields to
the state inside it, so `state.data` or `state.additional_carry` can be read at any level.

### Observation history

{class}`~loco_mujoco.core.wrappers.mjx.NStepWrapper` has two modes:

| Call | Observation |
|---|---|
| `NStepWrapper(env, n_steps)` | The whole observation of the last `n_steps` steps, oldest first, with a size of `n_steps * obs_dim`. At reset the older slots are zero. `include_actions=True` appends the action to every slot. |
| `NStepWrapper(env, n_steps, group="policy")` | The current observation unchanged, followed by the entries of one [observation group](observations.md#observation-groups) from the `n_steps - 1` previous steps, newest first. At reset the history is filled with the current values. |

The second mode keeps the indices of the current observation valid. A critic can go on reading its own
group from the head of the vector while only the actor consumes the history. PPO builds this from the
configuration keys `len_obs_history` and `obs_history_group`.

### Rollout wrappers

Two helpers collect rollouts for tests and quick evaluations.
{class}`~loco_mujoco.core.wrappers.rollout.MjxRolloutWrapper` runs episodes of an MJX environment with
`jax.lax.scan` and maps them over a batch of random keys, and
{class}`~loco_mujoco.core.wrappers.rollout.RolloutWrapper` does the same sequentially with the CPU step
functions. Both take an optional policy function and sample random actions without one.

```python
from loco_mujoco.core.wrappers import MjxRolloutWrapper

rollout = MjxRolloutWrapper(env)
obs, action, reward, next_obs, absorbing, done, cum_return = rollout.batch_rollout(keys, 100, None)
```

Each returned array has the shape (number of keys, number of steps, ...).

## See also

- [Simulation backends](backends.md) for the native CPU and MJX step functions.
- [Task factories](task_factories.md) for the arguments that `gym.make` forwards.
- [Training](../training/index.md) for the algorithms that use the MJX wrappers.
- API: {mod}`loco_mujoco.core.wrappers.gymnasium`, {mod}`loco_mujoco.core.wrappers.mjx`,
  {mod}`loco_mujoco.core.wrappers.rollout`.
