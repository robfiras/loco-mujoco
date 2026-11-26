# PPO

Proximal Policy Optimization (PPO, [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is an on-policy
actor-critic method. It collects a batch of experience with the current policy and improves the policy with
a clipped surrogate objective, which limits how far one update can move it. PPO is the default algorithm in
LocoMuJoCo: use it whenever you have a reward function, and expect it to be the most reliable choice with
thousands of parallel environments.

{class}`~loco_mujoco.algorithms.ppo_jax.PPOJax` is also the base of several other algorithms.
[DeepMimic-style tracking](deepmimic.md) is PPO with a tracking reward, [GAIL and AMP](amp_gail.md) use the
same policy update with a learned reward, and three of the [experimental algorithms](experimental.md) subclass
it.

## The example

`examples/training_examples/jax_rl` trains a Unitree Go2 quadruped to follow velocity commands. No motion
capture data is involved.

| Component | Choice | Purpose |
|---|---|---|
| Task factory | {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` | Creates the environment without a trajectory. |
| Environment | `MjxUnitreeGo2` | The MJX version of the Go2. |
| Goal | {class}`~loco_mujoco.core.observations.goals.GoalRandomRootVelocity` | Samples a target velocity in x, y and yaw and appends it to the observation. |
| Reward | {class}`~loco_mujoco.core.reward.default.LocomotionReward` | Rewards tracking the target velocity and penalises joint accelerations, torques, action rate and energy. |
| Termination | {class}`~loco_mujoco.core.terminal_state_handler.height.HeightBasedTerminalStateHandler` | Ends the episode when the root leaves a height range. |

### Train

```bash
cd examples/training_examples/jax_rl
python experiment.py
```

The script trains for 100 million environment steps with 2048 parallel environments; the example README
quotes about 20 minutes on an RTX 3080 Ti. It then

1. saves the agent as `PPOJax_saved.pkl` in the run directory `outputs/<date>/<time>/`,
2. sends the training curves to Weights & Biases (project `unitreeGo2`),
3. records a 200-step rollout of 20 environments to `LocoMuJoCo_recordings/` and uploads the video.

Any value of the configuration can be changed on the command line, for example

```bash
python experiment.py experiment.num_envs=4096 experiment.total_timesteps=2e8 experiment.n_seeds=3
```

See [Configuring experiments](configuration.md) for overrides, sweeps, and running without a Weights & Biases
account.

### Evaluate

```bash
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl --use_mujoco
```

`eval.py` rebuilds the environment from the configuration stored in the checkpoint, opens the viewer, and
runs the stochastic policy for 10,000 steps while recording a video. The first command simulates with MJX, the
second with MuJoCo on the CPU. A policy that works in MJX can be less robust in MuJoCo because the two
simulators differ slightly; [domain randomization](../user_guide/domain_randomization.md) during training
reduces that gap.

## Configuration walkthrough

The configuration has two top-level groups, `wandb` and `experiment`. Inside `experiment`, the entries
`task_factory` and `env_params` describe the environment and everything else is a hyperparameter of PPO:

```{literalinclude} ../../examples/training_examples/jax_rl/conf.yaml
:language: yaml
:start-at: "wandb:"
:end-at: "energy_coeff"
```

`task_factory.name` selects the factory and `env_params` are the keyword arguments of its `make` method. A
module of the environment is chosen with `<module>_type` and configured with `<module>_params`, here for the
goal, the reward and the terminal state handler. [Configuring experiments](configuration.md#the-environment)
lists all of them.

## Hyperparameters

Defaults are the values of `jax_rl/conf.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `num_envs` | 2048 | Number of parallel environments. |
| `num_steps` | 50 | Steps per environment between two updates. One update uses `num_envs * num_steps` transitions. |
| `total_timesteps` | 1e8 | Total number of environment steps. The number of updates is `total_timesteps / (num_envs * num_steps)`. |
| `update_epochs` | 4 | Passes over the collected batch per update. |
| `num_minibatches` | 32 | Minibatches per epoch. `num_envs * num_steps` must be divisible by it. |
| `lr` | 1e-4 | Initial learning rate of AdamW. |
| `desired_kl` | 0.01 | Target KL divergence of the adaptive learning rate, see below. Remove the key or set it to `null` for a constant learning rate. |
| `anneal_lr` | false | Linear decay of the learning rate to zero over the run. If true, `desired_kl` is not used. |
| `weight_decay` | 0.0 | Weight decay of AdamW. |
| `max_grad_norm` | 0.5 | Global gradient-norm clipping. |
| `gamma` | 0.99 | Discount factor. |
| `gae_lambda` | 0.95 | $\lambda$ of generalized advantage estimation. |
| `clip_eps` | 0.2 | Clipping range $\epsilon$ of the policy ratio and of the value update. |
| `vf_coef` | 0.5 | Weight of the value loss. |
| `ent_coef` | 0.0 | Weight of the entropy bonus. |
| `hidden_layers` | [512, 256] | Hidden layer sizes, used for both the actor and the critic. |
| `activation` | tanh | Activation function; any function name of `flax.linen`, such as `tanh`, `relu`, `elu` or `silu`. |
| `init_std` | 0.8 | Initial standard deviation of the Gaussian policy. |
| `learnable_std` | true | Whether the standard deviation is trained. With `false` it stays at `init_std`. |
| `normalize_env` | true | Normalize rewards by a running estimate of the standard deviation of the discounted return. |
| `n_seeds` | 1 | Number of seeds trained in parallel with `jax.vmap`. Seed `i` uses `jax.random.PRNGKey(i)`, starting at 1. |
| `debug` | false | Print episodic returns during training and skip the upload of training curves. |
| `validation.*` | inactive | Periodic validation rollouts, see [Evaluation](evaluation.md). |

The loss minimised in each minibatch is the usual PPO objective,

$$
L = -\,\mathbb{E}\big[\min\big(\rho_t \hat{A}_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\big]
    + c_v\, L_V - c_e\, \mathbb{E}\big[\mathcal{H}(\pi(\cdot \mid s_t))\big],
$$

with the probability ratio $\rho_t = \pi(a_t \mid s_t) / \pi_\text{old}(a_t \mid s_t)$, advantages $\hat{A}_t$
from GAE that are standardized per minibatch, the clipped value loss $L_V$, and the weights $c_v$ = `vf_coef` and
$c_e$ = `ent_coef`. Value targets are bootstrapped through time limits and set to zero only in absorbing states,
so a truncated episode does not look like a failure.

**Adaptive learning rate.** With `desired_kl` set, the learning rate is adjusted after every epoch from the
mean KL divergence between the old and the new policy: it is divided by 1.5 when the KL exceeds twice the
target and multiplied by 1.5 when it falls below half the target, within the bounds $10^{-5}$ and $10^{-2}$.

**Logged quantities.** Each update produces one entry of
`PPOSummaryMetrics`: mean episode return and length, value and actor loss, entropy, approximate KL, clip
fraction and learning rate. They appear under `Training/` in Weights & Biases.

## Network and policy options

The network is {class}`~loco_mujoco.algorithms.common.networks.ActorCritic`:

- The observation passes through a {class}`~loco_mujoco.algorithms.common.networks.RunningMeanStd` layer that
  normalizes it with running statistics. The statistics are part of the agent state and are saved with it.
- The actor and the critic are two separate multilayer perceptrons
  ({class}`~loco_mujoco.algorithms.common.networks.FullyConnectedNet`) with the layer sizes of `hidden_layers`
  and orthogonal initialization.
- The policy is a diagonal Gaussian. Its mean is the actor output; its log standard deviation is one
  state-independent parameter per action dimension.

{class}`~loco_mujoco.algorithms.common.policies.PPOPolicy` wraps the network for sampling actions and
evaluating log-probabilities.

Three optional keys of `experiment` change what the networks see. They are not part of the example
configuration; add them with a leading `+` on the command line or write them into the file.

| Key | Meaning |
|---|---|
| `actor_obs_group` | Name of an observation group, or a list of names. The actor only receives these observations. |
| `critic_obs_group` | The same for the critic. Together the two keys give an asymmetric actor-critic, for example a critic that sees privileged, noise-free observations. |
| `len_obs_history` | If larger than 1, the policy receives the last `len_obs_history` observations stacked. With `obs_history_group` only that group is stacked and only the actor receives the history. |

Observation groups are assigned to observations in the observation specification of the environment, see
[Observations](../user_guide/observations.md).

## Using the algorithm from Python

The example script is a thin wrapper. The same calls work in your own code; Hydra is not required, any
OmegaConf configuration with the same structure will do:

```python
import os
import jax
from omegaconf import OmegaConf

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

config = OmegaConf.load("examples/training_examples/jax_rl/conf.yaml")

# environment
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# agent
agent_conf = PPOJax.init_agent_conf(env, config)
rng = jax.random.PRNGKey(0)
agent_state = PPOJax.init_agent_state(env, agent_conf, rng)

# train
train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))
out = train_fn(rng, agent_state, traj)
agent_state = out["agent_state"]
print(out["training_metrics"].mean_episode_return[-1])

# save, load, play
os.makedirs("checkpoints", exist_ok=True)
path = PPOJax.save_agent("checkpoints", agent_conf, agent_state)   # checkpoints/PPOJax_saved.pkl
agent_conf, agent_state = PPOJax.load_agent(path)
PPOJax.play_policy(env, agent_conf, agent_state, n_envs=20, n_steps=200, deterministic=True, record=True)
```

A few details:

- `init_agent_conf` adds derived values to `config.experiment`, among them `num_updates`, `minibatch_size`
  and `validation_interval`.
- `save_agent` expects an existing directory and returns the path of the file it wrote. The file contains the
  configuration, the parameters, the optimizer state and the normalization statistics, so `load_agent` needs
  nothing else.
- `play_policy` rolls out `n_envs` MJX environments. `deterministic=True` plays the mean action,
  `record=True` writes a video whose path is then available as `env.video_file_path`. For an agent trained with
  several seeds, pass `train_state_seed` to choose one. `play_policy_mujoco` runs a single environment with
  the MuJoCo backend instead.
- Passing the returned `agent_state` to the training function again continues training, see
  [Checkpoints and resuming](configuration.md#checkpoints-and-resuming).
- `build_eval_fn` returns a function with the same signature that only performs evaluation rollouts, see
  [Evaluation](evaluation.md#calling-the-metrics-from-your-own-code).
