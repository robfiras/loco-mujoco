# Training

LocoMuJoCo ships ten learning algorithms written in JAX and Flax, and a set of runnable training examples
built on them. This section explains how training is organised, helps you choose an algorithm, and documents
each one together with its example.

All examples live in `examples/training_examples/`. Each folder holds an `experiment.py`, a Hydra configuration
`conf.yaml`, and usually an `eval.py`. The examples log to [Weights & Biases](https://wandb.ai), which is not
installed with LocoMuJoCo:

```bash
pip install wandb
```

## How training works

An MJX environment is a pure JAX function, and so are the algorithms. Rollout collection, advantage
computation and gradient updates are written with `jax.lax.scan` and compiled together into **one function**
that runs entirely on the GPU. A complete training run is a single call to that function. There is no Python
loop over environment steps, which is where the speed comes from.

Every algorithm derives from
{class}`~loco_mujoco.algorithms.common.base_algorithm.JaxRLAlgorithmBase` and exposes the same class methods.
Algorithms are never instantiated; all state is passed explicitly:

| Step | Call | Result |
|---|---|---|
| 1 | `init_agent_conf(env, config)` | The static part of the agent: the configuration, the network definitions and the optimizer. |
| 2 | `init_agent_state(env, agent_conf, rng)` | The dynamic part: network parameters, optimizer state and running statistics. A JAX pytree. |
| 3 | `build_train_fn(env, agent_conf, mh=None)` | A function `train_fn(rng, agent_state, traj)`. `mh` is an optional [metrics handler](evaluation.md). |
| 4 | `jax.jit(train_fn)(rng, agent_state, traj)` | Runs training. Returns a dictionary with `agent_state`, `training_metrics` and `validation_metrics`. |
| 5 | `save_agent(path, agent_conf, agent_state)` | Writes `<ClassName>_saved.pkl` into the directory `path`. |
| 6 | `load_agent(file)` and `play_policy(env, agent_conf, agent_state, ...)` | Restores an agent and rolls it out, optionally recording a video. |

The core of the PPO example shows steps 1 to 4:

```{literalinclude} ../../examples/training_examples/jax_rl/experiment.py
:language: python
:start-at: factory = TaskFactory
:end-at: out = train_fn
:dedent: 8
```

Three consequences of this design are worth knowing before you start:

- **Seeds are vectorized.** Because the training function is pure, `jax.vmap` runs several seeds in parallel
  on one GPU. The on-policy examples do this when `experiment.n_seeds` is larger than one; the saved agent then
  holds the parameters of all seeds.
- **Metrics arrive at the end.** The training function returns the metrics of all updates as arrays. The
  example scripts send them to Weights & Biases after the call returns, so curves appear when training has
  finished, not while it runs. Scripts that train in chunks (see [resuming](configuration.md#checkpoints-and-resuming))
  log after every chunk.
- **The first call compiles.** Expect a delay before training starts while JAX traces and compiles the whole
  loop.

The trajectory `traj` is an argument of the training function rather than a constant inside it. It can
therefore be replaced between calls without recompiling, which the
[distillation example](distillation.md#several-teachers-and-trajectories) uses.

## Algorithm overview

| Algorithm | Kind | Needs demonstrations | Policy update | Class | Example folder | Page |
|---|---|---|---|---|---|---|
| PPO | RL | no | on-policy | {class}`~loco_mujoco.algorithms.ppo_jax.PPOJax` | `jax_rl` | [PPO](ppo.md) |
| DeepMimic-style tracking | motion tracking | yes | on-policy (PPO) | {class}`~loco_mujoco.algorithms.ppo_jax.PPOJax` with {class}`~loco_mujoco.core.reward.trajectory_based.MimicReward` | `jax_rl_mimic` | [DeepMimic](deepmimic.md) |
| GAIL | adversarial imitation | yes | on-policy (PPO) | {class}`~loco_mujoco.algorithms.gail_jax.GAILJax` | `jax_gail` | [AMP and GAIL](amp_gail.md) |
| AMP | adversarial imitation with a task reward | yes | on-policy (PPO) | {class}`~loco_mujoco.algorithms.amp_jax.AMPJax` | `jax_amp` | [AMP and GAIL](amp_gail.md) |
| SuperMimic | motion tracking | yes | model-based, with a replay buffer | {class}`~loco_mujoco.algorithms.supermimic_jax.SuperMimicJax` | `jax_rl_mimic_supermimic` | [SuperMimic](supermimic.md) |
| SAC | RL | no | off-policy | {class}`~loco_mujoco.algorithms.experimental.sac_jax.SACJax` | `jax_rl_mimic_sac` | [Experimental](experimental.md#sac) |
| TD3 | RL | no | off-policy | {class}`~loco_mujoco.algorithms.experimental.td3_jax.TD3Jax` | `jax_rl_mimic_td3` | [Experimental](experimental.md#td3) |
| S2PG | RL with a stateful policy | no | on-policy (PPO) | {class}`~loco_mujoco.algorithms.experimental.s2pg_ppo_jax.S2PGPPOJax` | `jax_rl_mimic_s2pg` | [Experimental](experimental.md#s2pg) |
| BPTT-PPO | RL with a recurrent policy | no | on-policy (PPO) | {class}`~loco_mujoco.algorithms.experimental.bptt_ppo_jax.BPTTPPOJax` | `jax_rl_mimic_bptt` | [Experimental](experimental.md#bptt-ppo) |
| History-aware PPO | RL with an observation history | no | on-policy (PPO) | {class}`~loco_mujoco.algorithms.experimental.history_ppo_jax.HistoryPPOJax` | `jax_rl_mimic_history` | [Experimental](experimental.md#history-aware-ppo) |
| DAgger distillation | distillation | a teacher policy | supervised, with a replay buffer | {class}`~loco_mujoco.algorithms.experimental.vanilla_dagger_jax.VanillaDaggerJax` | `jax_distill` | [Distillation](distillation.md) |

The table has eleven rows for ten classes: DeepMimic-style tracking is not a separate algorithm but PPO trained
on a tracking reward. The algorithms marked "RL" optimise whatever reward the environment returns. Their
examples use the tracking reward of the DeepMimic example, so that all of them can be compared on the same task,
but they work equally well with a plain task reward.

The four core modules are in {mod}`loco_mujoco.algorithms`, the remaining six in
{mod}`loco_mujoco.algorithms.experimental`.

## Which one should I use

- **You have a reward and no motion data.** Use [PPO](ppo.md). It is the most tested algorithm in the library
  and the basis of most others.
- **You want a robot to reproduce reference motions as closely as possible.** Start with
  [DeepMimic-style tracking](deepmimic.md). It is PPO with a dense per-step tracking reward, needs no extra
  machinery, and is the recipe used for the benchmark results across all humanoids.
- **You want one tracking policy for many clips, or you train for deployment on a robot.** Look at
  [SuperMimic](supermimic.md). It learns a world model and improves the policy by differentiating through
  imagined rollouts.
- **You want a task solved in the style of a motion dataset, without frame-by-frame tracking.** Use
  [AMP](amp_gail.md): a task reward, for example velocity tracking, combined with a learned style reward.
- **You have demonstrations but no reward at all.** Use [GAIL](amp_gail.md). Expect lower tracking accuracy
  than with a tracking reward; it is included mainly as a baseline.
- **You need a policy with memory**, for example under partial observability or heavy domain randomization.
  Try [history-aware PPO](experimental.md#history-aware-ppo) first, then
  [S2PG](experimental.md#s2pg) or [BPTT-PPO](experimental.md#bptt-ppo).
- **You want an off-policy baseline.** Use [SAC](experimental.md#sac) or [TD3](experimental.md#td3).
- **You have trained policies and want a smaller, a single, or a differently observing one.** Use
  [DAgger distillation](distillation.md).

## Pages in this section

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} PPO
:link: ppo
:link-type: doc

Plain reinforcement learning: a Unitree Go2 learns to follow velocity commands.
:::

:::{grid-item-card} DeepMimic-style tracking
:link: deepmimic
:link-type: doc

PPO with a tracking reward and a reference-motion goal. The default imitation recipe.
:::

:::{grid-item-card} AMP and GAIL
:link: amp_gail
:link-type: doc

Adversarial imitation: a discriminator replaces or complements the reward.
:::

:::{grid-item-card} SuperMimic
:link: supermimic
:link-type: doc

Model-based motion tracking through a learned, differentiable world model.
:::

:::{grid-item-card} Experimental algorithms
:link: experimental
:link-type: doc

SAC, TD3, S2PG, BPTT-PPO and history-aware PPO.
:::

:::{grid-item-card} Distillation
:link: distillation
:link-type: doc

DAgger: compress one or several teacher policies into a student.
:::

:::{grid-item-card} Configuring experiments
:link: configuration
:link-type: doc

The anatomy of a `conf.yaml`, overrides, sweeps, logging, checkpoints and memory.
:::

:::{grid-item-card} Evaluation and metrics
:link: evaluation
:link-type: doc

The `eval.py` scripts, trajectory similarity measures and video recording.
:::
::::

```{toctree}
:hidden:
:caption: Core algorithms
:maxdepth: 1

ppo
deepmimic
amp_gail
supermimic
```

```{toctree}
:hidden:
:caption: More algorithms
:maxdepth: 1

experimental
distillation
```

```{toctree}
:hidden:
:caption: Running experiments
:maxdepth: 1

configuration
evaluation
```
