# SuperMimic

SuperMimic ([paper](https://arxiv.org/abs/2610.28480)) is a model-based algorithm for whole-body motion
tracking. It solves the same task as [DeepMimic-style tracking](deepmimic.md), a policy that follows reference
motions, with a different learning signal. PPO estimates the policy gradient from sampled returns. SuperMimic
learns a differentiable model of the robot's dynamics and computes the gradient of the return *through* that
model.

Choose SuperMimic when you want one policy to track many clips, or when you train a tracking policy for a real
robot. It has more components and more hyperparameters than PPO. For a first experiment on a single clip,
PPO with a tracking reward is the simpler starting point.

The implementation is {class}`~loco_mujoco.algorithms.supermimic_jax.SuperMimicJax`.

## How it works

Training alternates three steps.

1. **Collect.** All environments take one step with the deterministic policy plus Gaussian action noise. The
   transition is stored in a replay buffer.
2. **Fit the world model.** A neural network learns to predict the next simulator state `(qpos, qvel)` from
   the current state and the action. It is trained on short windows from the buffer, feeding its own
   predictions back in, so that it stays accurate over several steps and not only one.
3. **Improve the policy.** Starting from recently visited states, the policy is rolled out for
   `imag_horizon` steps *inside the world model*. The discounted sum of rewards along this imagined rollout,
   plus a learned value at its end, is differentiated with respect to the policy parameters by backpropagating
   through the whole rollout. A critic trained on TD($\lambda$) returns provides the value.

What distinguishes this implementation is how an imagined step is computed. It is an ordinary environment
step in which only the physics is replaced by the learned dynamics. The predicted state is passed through
MuJoCo's kinematics, and then the observations, the reward, the trajectory bookkeeping and the termination
check are evaluated by the **environment's own functions**. Nothing about the robot, the observation or the
reward is coded into the algorithm, so it works with other robots, goals and rewards without changes.

Some properties follow from this construction:

- **World-model state.** The model predicts MuJoCo's `(qpos, qvel)`. The robot must have a free root joint
  followed only by hinge or slide joints, which holds for all floating-base robots in LocoMuJoCo. The network
  sees heading-invariant features and outputs a correction on top of an explicit integration step; at
  initialization it extrapolates at constant velocity.
- **Kinematic rewards only.** Only kinematic quantities are recomputed during imagination. Reward or
  observation terms that read contacts or actuator forces keep the value of the state the rollout started
  from and contribute no gradient.
- **Termination.** If the terminal state handler provides a termination probability, as
  {class}`~loco_mujoco.core.terminal_state_handler.traj.RootTrackingTerminalStateHandler` does, it is used
  directly. Otherwise a small head of the world model learns to predict termination from the observation. The
  probability weights the imagined rewards, so the policy also receives a gradient that steers it away from
  falling.
- **Domain randomization.** With a domain randomizer, the randomized parameters of each episode are given to
  the world model and the critic as an additional context vector. The policy does not see them.
- **Backends.** The real environment steps run on MJX or, with `use_mjwarp: true`, on MuJoCo Warp. The
  imagined rollouts always run in JAX, because the gradient has to pass through the kinematics.

## The example

`examples/training_examples/jax_rl_mimic_supermimic` trains a Unitree G1 on the LAFAN1 clip `walk1_subject5`
with the same goal ({class}`~loco_mujoco.core.observations.goals.GoalTrajMimic`) and reward
({class}`~loco_mujoco.core.reward.trajectory_based.MimicReward`) as the DeepMimic example. The only
difference in the environment is the termination: the episode ends when the root deviates too far from the
*current* reference frame, which stays meaningful when the dataset contains motions on the ground.

```{important}
On the GPU, SuperMimic's training function needs `jax<=0.10.1,!=0.10.0` to compile. This is the version range
LocoMuJoCo declares as a dependency, so a standard installation satisfies it. Keep the constraint when you
install the CUDA build of JAX:

    pip install "jax[cuda12]<=0.10.1,!=0.10.0"
```

### Train

```bash
cd examples/training_examples/jax_rl_mimic_supermimic
python experiment.py
```

This trains with 4096 environments for 1.31 billion steps, saves `SuperMimicJax_saved.pkl` to the run directory,
logs to the Weights & Biases project `deepmimic_supermimic`, and records a short video. The training metrics
include the world-model loss and prediction quality, the imagined return and the critic loss.

Other clips and a shorter run:

```bash
python experiment.py \
    'experiment.task_factory.params.lafan1_dataset_conf.dataset_name=[walk1_subject1,dance2_subject4]' \
    experiment.total_timesteps=3e8
```

The example trains one seed; `n_seeds` has no effect in this script.

### Evaluate

```bash
python eval.py --path outputs/<date>/<time>/SuperMimicJax_saved.pkl
python eval.py --path outputs/<date>/<time>/SuperMimicJax_saved.pkl --use_mujoco
```

`eval.py` plays the deterministic policy with the viewer, in MJX or in MuJoCo, and records a video.

`eval_clips.py` produces numbers instead of a video. It plays every clip of the dataset once from its first
frame, without action noise and with domain randomization removed, and reports per clip

- the **normalized return**: the undiscounted sum of rewards divided by the maximum attainable sum,
- the number of completed steps, and
- whether the clip **failed**, meaning the rollout terminated before the end of the clip.

A summary line gives the mean normalized return, the number of failed clips and the share of completed
frames. The script takes the run directory and the checkpoint file name, and expects the run's `config.yaml` in
the run directory. Hydra stores that file in the `.hydra` subfolder, so copy it first:

```bash
cp outputs/<date>/<time>/.hydra/config.yaml outputs/<date>/<time>/
python eval_clips.py outputs/<date>/<time> SuperMimicJax_saved.pkl
```

The script reads the clip names from `lafan1_dataset_conf`, so it applies to runs on LAFAN1. Environments or
modules defined outside LocoMuJoCo can be registered with `--import=<module>`.

## Configuration

The keys are grouped as in `conf.yaml`. Values are the defaults of the example.

### Observation routing

| Key | Default | Meaning |
|---|---|---|
| `actor_obs_group` | full observation | Observation group(s) the policy receives. |
| `critic_obs_group` | full observation | Observation group(s) the critic receives, for example privileged, noise-free observations. |
| `termination_obs_group` | full observation | Observation group(s) for the learned termination head. Must contain the reference for tracking terminations. |

### Policy and critic

| Key | Default | Meaning |
|---|---|---|
| `hidden_layers` | [2048, 1024, 512, 256] | Policy network. |
| `critic_hidden_layers` | [256, 128] | Critic network. |
| `activation` | silu | Activation of both. |
| `lr_actor`, `lr_critic` | 2e-4 | Learning rates. |
| `actor_max_grad_norm`, `actor_max_grad_value` | 1.0, 0.1 | Clipping of the policy gradient by global norm and per element. |
| `actor_grad_spike_factor` | 10.0 | Skip a policy update whose gradient norm exceeds this multiple of its running average. 0 disables the check. |
| `critic_max_grad_norm` | 1.0 | Gradient clipping of the critic. |
| `critic_epochs`, `critic_minibatches` | 5, 4 | Critic passes and minibatches per update. |
| `action_noise_std` | 0.1 | Standard deviation of the exploration noise added to the deterministic policy. |

### World model

| Key | Default | Meaning |
|---|---|---|
| `wm_hidden_layers` | [2048, 2048, 2048] | Dynamics network. |
| `wm_term_hidden_layers` | [64, 64] | Termination head. |
| `wm_activation` | silu | Activation. |
| `lr_wm`, `wm_max_grad_norm` | 2e-4, 1.0 | Learning rate and gradient clipping. |
| `wm_window` | 8 | Length of the windows the model is unrolled over during training. |
| `wm_batch_size`, `wm_batches_per_update` | 2048, 10 | Windows per batch and batches per update. |
| `wm_term_loss_weight` | 1.0 | Weight of the termination loss. |
| `wm_in_scale`, `wm_out_scale`, `wm_err_scale` | see file | Scales of the input features, of the predicted corrections, and of the state errors in the loss, per group of state variables. |
| `buffer_len` | 1000 | Length of the replay buffer in environment steps; it holds `buffer_len * num_envs` transitions. |
| `max_abs_qvel` | 100.0 | States with larger velocities, or non-finite values, are treated as simulator failures and kept out of the buffers. |

### Imagination

| Key | Default | Meaning |
|---|---|---|
| `imag_horizon` | 32 | Length of an imagined rollout. |
| `imag_batch_size` | 4096 | Number of imagined rollouts per policy update. |
| `start_buffer_len` | 16 | Imagined rollouts start from states of the last `start_buffer_len` environment steps. |
| `gamma`, `gae_lambda` | 0.99, 0.95 | Discount factor and $\lambda$ of the TD($\lambda$) returns. |
| `remat` | true | Recompute each imagined step in the backward pass instead of storing it. Saves memory at the cost of compute. |
| `net_dtype` | float32 | Data type of the matrix products in the networks. `bfloat16` is faster; the parameters stay in float32. |

### Schedule

| Key | Default | Meaning |
|---|---|---|
| `num_envs` | 4096 | Parallel environments. |
| `total_timesteps` | 1.31e9 | Total environment steps: 320,000 steps of 4096 environments. |
| `updates_per_env_step` | 0.4 | Update-to-data ratio: updates of world model, policy and critic per vectorized environment step. |
| `warmup_steps` | 40 | Environment steps collected before the first update. |
| `lr_final_ratio`, `lr_decay_fraction` | 0.03, 0.78 | Cosine decay of all learning rates to `lr_final_ratio` times the initial value over the first `lr_decay_fraction` of training. |
| `log_every` | 50 | Environment steps averaged into one logged entry. |

The source file documents further options that are off by default.

## On a real robot

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) contains the SuperMimic training setups that
are deployed on hardware, for the Unitree G1 and the AgiBot X2 Ultra (`training/<robot>/train_supermimic.py`),
together with the tools to take a checkpoint to the robot. They differ from this example in the points that
matter for transfer: they train one policy on 40 LAFAN1 clips, use MuJoCo Warp with 4096 environments, enable
self-collisions, and randomize the domain with pushes, sensor noise and bias, friction and a centre-of-mass
offset. Noise is applied only to the policy's observations; the critic receives clean values.

The example on this page is the same algorithm reduced to a minimal setup. To move towards the deployment
configuration, enable a randomizer under `env_params` (the example file contains a commented block) and route
the observations with the three `*_obs_group` keys. See [Sim-to-real](../sim2real/index.md) for the
deployment workflow.
