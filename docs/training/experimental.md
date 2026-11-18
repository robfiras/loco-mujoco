# Experimental algorithms

The package {mod}`loco_mujoco.algorithms.experimental` contains five further reinforcement learning algorithms:
two off-policy methods (SAC and TD3) and three ways of giving a PPO policy memory (S2PG, BPTT-PPO and
history-aware PPO).

```{warning}
These algorithms are experimental. They are less tested than the [core algorithms](index.md), their example
configurations are starting points and not tuned results, and their interfaces and configuration keys may
change between releases. Verify them on your task before relying on them.
```

All five follow the common interface described in the [training overview](index.md#how-training-works) and are
imported from the subpackage:

```python
from loco_mujoco.algorithms.experimental import SACJax, TD3Jax, S2PGPPOJax, BPTTPPOJax, HistoryPPOJax
```

Every example trains a Unitree G1 on the LAFAN1 clip `walk1_subject5` with the tracking reward of the
[DeepMimic example](deepmimic.md), so the algorithms can be compared directly with PPO on the same task. None of
them depends on that task; they optimise whatever reward the environment returns.

| Algorithm | Class | Example folder | Scripts |
|---|---|---|---|
| [SAC](#sac) | {class}`~loco_mujoco.algorithms.experimental.sac_jax.SACJax` | `jax_rl_mimic_sac` | `experiment.py`, `experiment_resume.py`, `eval.py` |
| [TD3](#td3) | {class}`~loco_mujoco.algorithms.experimental.td3_jax.TD3Jax` | `jax_rl_mimic_td3` | `experiment.py`, `eval.py` |
| [S2PG](#s2pg) | {class}`~loco_mujoco.algorithms.experimental.s2pg_ppo_jax.S2PGPPOJax` | `jax_rl_mimic_s2pg` | `experiment.py` |
| [BPTT-PPO](#bptt-ppo) | {class}`~loco_mujoco.algorithms.experimental.bptt_ppo_jax.BPTTPPOJax` | `jax_rl_mimic_bptt` | `experiment.py` |
| [History-aware PPO](#history-aware-ppo) | {class}`~loco_mujoco.algorithms.experimental.history_ppo_jax.HistoryPPOJax` | `jax_rl_mimic_history` | `experiment.py` |

Each is started with `python experiment.py` from its folder and saves `<ClassName>_saved.pkl` to the run
directory. The examples without an `eval.py` record a video at the end of training; to replay such an agent
later, call `load_agent` and `play_policy` on its class as shown on the [PPO page](ppo.md#using-the-algorithm-from-python).

## Off-policy algorithms

SAC and TD3 share one engine, {class}`~loco_mujoco.algorithms.experimental.offpolicy_base.OffPolicyBase`. It
owns the replay buffer, the twin Q-networks with a slowly updated target copy, and the training loop. One
iteration of the loop steps all environments once, adds the `num_envs` transitions to the buffer, and then
performs `gradient_steps` updates on minibatches drawn from the buffer.

On-policy methods discard data after one update; off-policy methods reuse it many times. They are therefore
the methods to try when environment steps are expensive relative to gradient steps. With thousands of
cheap GPU environments that advantage shrinks, and on the tracking task PPO reaches a higher return than both
in our comparison on the Unitree H1. Treat them as baselines and as a basis for off-policy research.

The keys shared by both:

| Key | Example value | Meaning |
|---|---|---|
| `num_envs` | 2048 | Parallel environments; also the number of transitions added per iteration. |
| `total_timesteps` | 1e7 | Total environment steps. |
| `buffer_size` | 1,000,000 | Capacity of the replay buffer in transitions. |
| `learning_starts` | 1,000,000 | Number of transitions in the buffer before updates begin. |
| `batch_size` | 256 | Minibatch size of one update. |
| `gradient_steps` | 20 | Updates per iteration. Together with `num_envs` this sets the update-to-data ratio. |
| `gamma` | 0.99 | Discount factor. |
| `tau` | 0.005 | Rate of the soft update of the target critic. |
| `lr_actor`, `lr_critic` | 3e-4 | Learning rates. |
| `hidden_layers` | [256, 256] | Layer sizes of the actor and of each Q-network. |
| `max_grad_norm` | 0.5 | Gradient clipping. |
| `log_every` | 100 | Iterations averaged into one logged entry. |
| `use_obs_norm`, `normalize_env` | false | Running normalization of observations and rewards. |

```{note}
Keep `use_obs_norm` and `normalize_env` off, as the examples do. Transitions in the replay buffer were
normalized with the statistics of the time they were collected, while the critic and its target use the
current ones, which makes the Bellman targets inconsistent.
```

Both examples train a single seed. Both `eval.py` scripts accept `--path` and `--use_mujoco` like the
[PPO script](ppo.md#evaluate).

### SAC

Soft Actor-Critic ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) learns a stochastic policy that
maximises the return plus the entropy of the policy, weighted by a temperature $\alpha$. The actor is a
Gaussian with tanh squashing; $\alpha$ is adapted automatically so that the policy entropy stays near a
target. Try SAC when you want an off-policy method that explores by itself and is comparatively insensitive
to its hyperparameters.

| Key | Example value | Meaning |
|---|---|---|
| `init_alpha` | 0.01 | Initial temperature. |
| `learnable_alpha`, `lr_alpha` | true, 3e-4 | Whether $\alpha$ is adapted, and its learning rate. |
| `target_entropy` | -200.7 | Entropy the temperature update aims for. The default, if the key is absent, is minus the number of actions. |
| `init_std`, `learnable_std` | 0.1, true | Initial standard deviation of the policy and whether it is trained. |
| `state_dependent_std` | true | Standard deviation as a network output instead of one free parameter per action. |
| `log_std_min`, `log_std_max` | -20, 2 | Bounds of the log standard deviation. |

The default target entropy assumes a standard deviation near 1. For tracking, where a small action noise is
wanted, a lower target that matches the desired noise level works better: roughly the number of actions times
the logarithm of the desired standard deviation. The example sets such a low target.

`experiment_resume.py` with `conf_resume.yaml` trains in chunks of `timesteps_per_chunk` steps and logs after
each, in the same way as the [DeepMimic resume script](deepmimic.md#training-in-chunks-and-resuming). The replay
buffer is part of the agent state and is carried from chunk to chunk.

### TD3

Twin Delayed DDPG ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)) learns a deterministic policy.
Exploration comes from Gaussian noise added to the actions during data collection. Three measures counter the
overestimation of Q-values: two critics of which the smaller target is used, noise on the target action, and
actor updates that are less frequent than critic updates. Try TD3 when you want a deterministic policy, for
example as a counterpart to SAC. Its actor network is also the policy network of
[SuperMimic](supermimic.md).

| Key | Example value | Meaning |
|---|---|---|
| `exploration_noise` | 0.1 | Standard deviation of the action noise during data collection. |
| `target_noise` | 0.1 | Standard deviation of the noise added to the target action. |
| `noise_clip` | 0.3 | Bound of the target noise. |
| `policy_delay` | 1 | The actor and the target networks are updated every `policy_delay` critic updates. |
| `lambda_oob` | 0.1 | Weight of a penalty on actions outside the action bounds in the actor loss. |
| `output_squash` | not set (`none`) | `tanh` bounds the actor output to (-1, 1); by default the output is unbounded and the penalty above keeps it in range. |

## Policies with memory

A feed-forward policy acts on the current observation alone. When the observation does not determine the
state, for example with sensor noise, unobserved physical parameters under domain randomization, or terrain
the robot cannot see, a policy that remembers the past can do better. The three algorithms below add memory to
PPO in different ways. All subclass {class}`~loco_mujoco.algorithms.ppo_jax.PPOJax`, use its
[hyperparameters](ppo.md#hyperparameters), and support parallel seeds and the
[validation loop](evaluation.md) in their examples.

| | Memory | Gradient through time | Cost |
|---|---|---|---|
| History-aware PPO | A fixed window of past observations | not needed | lowest |
| S2PG | A learned stochastic internal state | no | moderate |
| BPTT-PPO | A learned recurrent state | yes, over the whole rollout | highest |

### History-aware PPO

The policy receives the last `history_len` observations, and optionally the last actions, instead of only the
current one. This is the simplest form of memory and often sufficient when a short context is enough, such as
estimating velocities or contact timing from recent observations. Try it first.

The history is encoded either by a multilayer perceptron or by a small Transformer. The implementation stores
single observations in the rollout buffer and assembles the windows per minibatch, so memory use does not grow
with the length of the history.

| Key | Example value | Meaning |
|---|---|---|
| `history_len` | 10 | Number of steps in the window. |
| `include_actions_in_history` | true | Add the past actions to each step of the window. |
| `encoder_type` | mlp | `mlp`: project each step, concatenate, apply a multilayer perceptron. `transformer`: project each step, apply self-attention blocks, average. |
| `n_features` | 256 | Width of the per-step projection. |
| `d_model`, `n_heads`, `n_layers` | 128, 4, 2 | Transformer width, attention heads and number of blocks. Only used with `encoder_type: transformer`. |

The example configuration sets a short run of 10 million steps; raise `total_timesteps` for a full training.

```{note}
`PPOJax` has a simpler built-in alternative: `len_obs_history` stacks past observations in front of the
standard network, see [network and policy options](ppo.md#network-and-policy-options).
```

### S2PG

The Stochastic Stateful Policy Gradient ([Al-Hafez et al., 2023](https://arxiv.org/abs/2311.04082)) gives the
policy an internal state $z$ and makes the transition of that state *stochastic*. The policy outputs a joint
distribution over the action and the next internal state, and PPO's probability ratio is taken over both.
The internal state is thereby trained with the ordinary policy gradient, without backpropagation through
time. This avoids the cost and the gradient instabilities of BPTT while keeping a learned memory.
Try it when a fixed history window is not enough and BPTT is too slow or unstable.

The network embeds the observation and the previous action, feeds them to a GRU or LSTM cell whose carry is
$z$, and takes the cell output as the mean of the next state. The critic receives the observation, $z$ and the
previous action.

| Key | Example value | Meaning |
|---|---|---|
| `hidden_state_dim` | 64 | Dimension of the internal state $z$. |
| `init_std_z` | 0.1 | Initial standard deviation of the state transition. |
| `rnn_type` | gru | `gru`, `lstm`, or `vanilla` for a plain multilayer perceptron instead of a recurrent cell. |
| `n_features` | 256 | Width of the embeddings before and after the cell. |
| `ema_coef` | not set (1.0) | Smoothing of the state transition; values below 1 blend the new state mean with the old state. |

### BPTT-PPO

BPTT-PPO uses a classical recurrent policy: a GRU or LSTM with a deterministic hidden state, trained by
backpropagation through time. During the update, the stored rollouts are replayed through the network from
their initial hidden state and the gradient flows through the entire sequence of `num_steps` steps. Try it
when the task needs memory over long horizons and you can afford the cost.

Because sequences must stay intact, minibatches are formed over environments and not over single transitions.
This changes the meaning of one PPO key:

| Key | Example value | Meaning |
|---|---|---|
| `num_minibatches` | 32 | Number of minibatches of whole sequences. `num_envs` must be divisible by it; each minibatch holds `num_envs / num_minibatches` sequences. |
| `num_steps` | 200 | Rollout length, and therefore also the length of backpropagation through time. |
| `hidden_state_dim` | 64 | Size of the recurrent state. |
| `rnn_type` | gru | `gru`, `lstm` or `vanilla`. |
| `n_features` | 256 | Width of the embeddings before and after the cell. |
| `ema_coef` | 1.0 | Smoothing of the hidden-state update; 1.0 means none. |

Memory and compute grow with `num_steps`, since the whole sequence is differentiated at once. Reduce
`num_steps` or `num_envs` if the GPU runs out of memory.
