# AMP and GAIL

Adversarial imitation learning replaces a hand-written imitation reward with a learned one. A
*discriminator* is trained to tell states visited by the policy from states of the demonstrations. The policy
is rewarded for states the discriminator takes for demonstrations, so it is pushed towards the distribution
of the dataset instead of towards a specific frame at a specific time.

LocoMuJoCo implements two variants:

- **GAIL** (Generative Adversarial Imitation Learning,
  [Ho and Ermon, 2016](https://arxiv.org/abs/1606.03476)) in
  {class}`~loco_mujoco.algorithms.gail_jax.GAILJax`.
- **AMP** (Adversarial Motion Priors, [Peng et al., 2021](https://arxiv.org/abs/2104.02180)) in
  {class}`~loco_mujoco.algorithms.amp_jax.AMPJax`.

Both use PPO for the policy update, so everything on the [PPO page](ppo.md) about networks and hyperparameters
applies.

## When to use which

| | GAIL | AMP |
|---|---|---|
| Typical use | Pure imitation: the demonstrations are the only training signal. | A task reward defines *what* to do, the discriminator defines the *style*. |
| Reward in the example | Discriminator only (`proportion_env_reward: 0.0`) | Half task reward, half discriminator (`proportion_env_reward: 0.5`) |
| Discriminator loss | Binary cross-entropy | Least squares |
| Reward from the discriminator | $-\log(1 - D(s))$, unbounded above | $\max\big(0,\ 1 - \tfrac{1}{4}(D(s) - 1)^2\big)$, between 0 and 1 |

Use AMP when you have a task that can be written as a reward, such as following a velocity command, and want
the robot to solve it with natural-looking motion. Use GAIL when there is no task reward. If you need the
robot to follow a particular clip in time, neither is the right tool: a
[tracking reward](deepmimic.md) is simpler and tracks considerably more accurately, because it gives the policy
an explicit target at every step. GAIL is included mainly as a baseline and as a starting point for
adversarial methods.

## How it works

One training iteration has three stages.

1. **Rollout.** The policy collects `num_steps` steps in `num_envs` environments, as in PPO.
2. **Policy update.** The reward of every collected transition is a mix of the environment reward
   $r_\text{env}$ and the discriminator reward $r_D$,

   $$
   r = p\, r_\text{env} + (1 - p)\, r_D(s),
   $$

   where $p$ is `proportion_env_reward`, and PPO runs its usual epochs on it. With `normalize_env: true` the environment reward is normalized before
   mixing.
3. **Discriminator update.** The discriminator takes `n_disc_epochs` gradient steps. Each step draws
   `disc_minibatch_size` observations from the rollout just collected and the same number from the
   demonstrations.

### Discriminator inputs

The discriminator receives a single observation $s$: the same vector the policy sees, including the goal. It
does not receive actions, so motion capture data without actions is sufficient. It is a
{class}`~loco_mujoco.algorithms.common.networks.FullyConnectedNet` with the layer sizes and activation of the
policy and its own running input normalization.

Because the discriminator judges whatever is in the observation, the observation defines what "imitating"
means. Quantities that the dataset cannot provide reliably, or that the policy should be free to choose,
should not be part of it.

### Losses and rewards

GAIL trains the discriminator output (after a sigmoid) $D(s) \in (0, 1)$ with the binary cross-entropy,
labelling demonstrations 1 and policy samples 0:

$$
L_D = -\,\mathbb{E}_{s \sim \text{demo}}\big[\log D(s)\big] - \mathbb{E}_{s \sim \pi}\big[\log(1 - D(s))\big]
      - c\,\mathcal{H}(D),
$$

where the last term is an optional entropy bonus with weight $c$ = `disc_ent_coef`. The reward is
$r_D(s) = -\log(1 - D(s))$.

AMP uses the raw output $D(s) \in \mathbb{R}$ with a least-squares loss towards the same targets, 1 for
demonstrations and 0 for policy samples, and the bounded reward
$r_D(s) = \max\big(0,\ 1 - \tfrac{1}{4}(D(s) - 1)^2\big)$. The bounded reward makes it easier to balance against
a task reward. `AMPJax` subclasses `GAILJax` and overrides only these two functions.

## How demonstrations are provided

The discriminator needs demonstrations as *observations*, in exactly the format the policy sees. The
environment builds them from its trajectory with
{meth}`~loco_mujoco.environments.base.LocoEnv.create_dataset`: it replays the motion capture data and
evaluates the observation functions on every frame. The result is attached to the agent configuration:

```python
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

expert_dataset = env.create_dataset()          # TrajectoryTransitions with .observations
agent_conf = GAILJax.init_agent_conf(env, config)
agent_conf = agent_conf.add_expert_dataset(expert_dataset)
```

Building the dataset takes a moment, so both example scripts cache it in the file `expert_traj.npz` in the
working directory and load it from there on later runs.

```{warning}
The cache is not tied to the configuration. After changing the robot, the dataset or the observations, delete
`expert_traj.npz`, otherwise the discriminator is trained on the demonstrations of the previous setup.
```

The demonstrations are not stored in the checkpoint. A loaded agent can be played directly; to continue
training it, create the dataset again and call `add_expert_dataset`.

Observations are computed from kinematic quantities of the mocap data. Quantities that depend on forces or
contacts are not reliable in such a dataset and are best left out of the observation, see
[Observations](../user_guide/observations.md).

## The examples

Both examples use the Unitree H1 and the `run` task of the native dataset.

| | `jax_gail` | `jax_amp` |
|---|---|---|
| Goal | {class}`~loco_mujoco.core.observations.goals.GoalTrajMimic`: the reference pose of the current step | {class}`~loco_mujoco.core.observations.goals.GoalTrajRootVelocity`: the root velocity of the reference |
| Environment reward | {class}`~loco_mujoco.core.reward.trajectory_based.MimicReward`, used for monitoring only | {class}`~loco_mujoco.core.reward.trajectory_based.TargetVelocityTrajReward`: match the root velocity in x, y and yaw |
| What the policy learns | To reproduce the running motion | To move at the commanded velocity in a running style |
| Validation with trajectory metrics | on | off |
| Saved file | `GAILJax_saved.pkl` | `AMPJax_saved.pkl` |

Both are minimal demonstrations of the algorithms and are not tuned for the best possible result.

### Train

```bash
cd examples/training_examples/jax_gail    # or jax_amp
python experiment.py
```

Each trains for 75 million steps with 2048 environments, which the example READMEs quote at roughly ten
minutes on an RTX 3080 Ti, saves the agent to `outputs/<date>/<time>/`, logs to the Weights & Biases project
`gail` or `amp`, and records a short video.

During training, watch `Discriminator/Output Policy` and `Discriminator/Output Expert`. These are the mean
discriminator outputs on policy samples and on demonstrations (probabilities for GAIL, raw outputs for AMP).
If they separate completely, the discriminator has won and the policy no longer receives a useful reward; if
they are identical from the start, the discriminator is not learning.

### Evaluate

```bash
python eval.py --path outputs/<date>/<time>/GAILJax_saved.pkl
python eval.py --path outputs/<date>/<time>/GAILJax_saved.pkl --use_mujoco
```

As for PPO, the first command plays the policy in MJX and the second in MuJoCo, both with the viewer and a
recording.

## Configuration

The examples share the PPO keys described on the [PPO page](ppo.md#hyperparameters). The adaptive learning
rate (`desired_kl`) is not available here; the learning rate is constant. The additional keys are:

| Key | `jax_gail` | `jax_amp` | Meaning |
|---|---|---|---|
| `proportion_env_reward` | 0.0 | 0.5 | Share $p$ of the environment reward in the policy's reward. With 0 the environment reward is still logged as the episode return but has no influence on training. |
| `disc_lr` | 5e-5 | 5e-5 | Learning rate of the discriminator. |
| `n_disc_epochs` | 10 | 50 | Discriminator gradient steps per policy update. |
| `disc_minibatch_size` | 2048 | 2048 | Policy samples and demonstration samples per discriminator step, each. |
| `disc_ent_coef` | 0.0 | 0.0 | Weight of the entropy bonus in the GAIL discriminator loss. Not used by AMP. |
| `num_steps` | 14 | 14 | Short rollouts, so that the discriminator is updated often. |
| `lr` | 1e-4 | 6e-5 | Learning rate of the policy. |
| `init_std` | 0.125 | 0.2 | Fixed standard deviation of the policy (`learnable_std: false`). |
| `clip_eps` | 0.2 | 0.1 | PPO clipping range. |
| `weight_decay` | 0.0 | 1e-4 | Weight decay for policy and discriminator. |

The balance between the two players is the main thing to tune. A discriminator that learns too fast
(high `disc_lr` or `n_disc_epochs`) saturates and stops giving gradients; one that learns too slowly gives a
reward that is easy to exploit. For AMP, `proportion_env_reward` trades task performance against style.

```{seealso}
[DeepMimic-style tracking](deepmimic.md) for imitation with an explicit tracking reward, and
[Evaluation](evaluation.md) for the trajectory metrics used to compare the approaches.
```
