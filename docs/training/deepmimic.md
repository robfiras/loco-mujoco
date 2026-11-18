# DeepMimic-style motion tracking

Motion tracking trains a policy to reproduce a reference motion as closely as possible, frame by frame.
LocoMuJoCo follows DeepMimic ([Peng et al., 2018](https://arxiv.org/abs/1804.02717)): the policy is told where
the reference is at the current step and is rewarded for being close to it. No new algorithm is needed for
this. The recipe is three existing parts:

- {class}`~loco_mujoco.algorithms.ppo_jax.PPOJax` as the learner,
- {class}`~loco_mujoco.core.reward.trajectory_based.MimicReward` as the reward, and
- {class}`~loco_mujoco.core.observations.goals.GoalTrajMimic` as the goal observation.

This is the default way to imitate motion capture data in LocoMuJoCo. Choose it when you want accurate
tracking of one or a few clips and prefer a simple, dense reward over a learned one. For many clips with one
policy, also consider [SuperMimic](supermimic.md).

```{figure} ../../imgs/unitree_h1_walk_anydir.gif
:alt: A Unitree H1 tracking walking motions in several directions
:width: 60%
:align: center

A Unitree H1 trained with this recipe on walking motions in several directions.
```

## The example

`examples/training_examples/jax_rl_mimic` trains a Unitree G1 on one LAFAN1 walking clip
(`walk1_subject5`). The dataset is selected in the `task_factory` block of `conf.yaml`:

```{literalinclude} ../../examples/training_examples/jax_rl_mimic/conf.yaml
:language: yaml
:start-at: "  task_factory:"
:end-at: "        - right_foot_mimic"
```

{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` loads the clips for the chosen robot
(LAFAN1 and the native dataset are downloaded and cached automatically) and returns the environment together
with the trajectory. See
[Datasets](../user_guide/datasets.md) for the available sources and
[Task factories](../user_guide/task_factories.md) for the dataset configurations.

### Train

```bash
cd examples/training_examples/jax_rl_mimic
python experiment.py
```

Training runs for 300 million steps with 2048 environments, saves `PPOJax_saved.pkl` to the run directory
`outputs/<date>/<time>/`, logs to the Weights & Biases project `deepmimic`, and records a short video. Training
longer gives smoother motion.

To track other motions, override the dataset. The first command uses two LAFAN1 clips; the second switches to
the Unitree H1 and the `walk` task of the native dataset:

```bash
python experiment.py 'experiment.task_factory.params.lafan1_dataset_conf.dataset_name=[walk1_subject1,dance2_subject4]'

python experiment.py experiment.env_params.env_name=MjxUnitreeH1 \
    '~experiment.task_factory.params.lafan1_dataset_conf' \
    '+experiment.task_factory.params.default_dataset_conf={task: walk}'
```

### Evaluate

```bash
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl --use_mujoco
```

The script rebuilds the environment from the checkpoint, replaces the goal by
{class}`~loco_mujoco.core.observations.goals.GoalTrajMimicv2`, which draws the reference with the robot's own
geometry, and plays the policy for 10,000 steps while recording. Without `--use_mujoco` the rollout runs
in MJX, with it in MuJoCo on the CPU. Policies trained without
[domain randomization](../user_guide/domain_randomization.md) can be less robust in MuJoCo than in MJX
because the two simulators differ slightly.

## The reward

`MimicReward` compares five quantities of the simulated robot with the reference frame of the current step.
Each difference $d_i$ is a mean squared error that is turned into a reward between 0 and 1 by an exponential
kernel, and the terms are added with weights:

$$
r = \max\Big(0,\ \sum_i w_i \exp(-\beta_i\, d_i) + r_\text{pen}\Big)
$$

| Term | Compared quantity | Weight $w_i$ | Example | Class default | Scale $\beta_i$ | Default |
|---|---|---|---|---|---|---|
| Joint positions | `qpos`, including the root pose | `qpos_w_sum` | 0.4 | 0.0 | `qpos_w_exp` | 10.0 |
| Joint velocities | `qvel`, including the root velocity | `qvel_w_sum` | 0.2 | 0.0 | `qvel_w_exp` | 2.0 |
| Site positions | positions of the mimic sites relative to the main site | `rpos_w_sum` | 0.5 | 0.5 | `rpos_w_exp` | 100.0 |
| Site orientations | orientations of the mimic sites relative to the main site | `rquat_w_sum` | 0.3 | 0.3 | `rquat_w_exp` | 10.0 |
| Site velocities | relative angular and linear velocities of the mimic sites, as two terms with the same weight | `rvel_w_sum` | 0.1 | 0.0 | `rvel_w_exp` | 0.1 |

*Mimic sites* are markers attached to body links such as the torso, hands and feet. Every humanoid defines a
default set; `sites_for_mimic` selects a subset, and the first entry (or `main_site_name`) is the main site
that the others are expressed relative to. Because these terms are relative to the robot's own body, they
reward the pose and not the absolute position in the world. `joints_for_mimic` restricts the joint terms to a
list of joints in the same way.

The penalty $r_\text{pen}$ is zero or negative and is bounded below by $-1$. It consists of a penalty for
actions outside the action bounds (`action_out_of_bounds_coeff`, default 0.01) and three optional
regularizers that are off by default: `joint_acc_coeff`, `joint_torque_coeff` and `action_rate_coeff`.

All of these keys go under `env_params.reward_params` in the configuration. The
[rewards page](../user_guide/rewards.md) covers the other reward classes.

## The goal observation

A tracking policy has to know what to track. `GoalTrajMimic` appends the reference state of the current step
to the observation:

- the reference joint positions, without the horizontal position of the root,
- the reference joint velocities, and
- for every mimic site other than the main one: position, orientation (as a rotation vector) and the
  six-dimensional velocity, relative to the main site.

Like the reward, the goal accepts `sites_for_mimic` and `main_site_name` in `goal_params`. With
`visualize_goal: true` the reference sites are drawn in the viewer.

The reference frame advances with the simulation. At the start of an episode the robot is placed on a frame
of the trajectory, and the episode ends when the robot's root leaves the range of root poses seen in the
dataset. Both defaults come from `ImitationFactory`; see
[Initial and terminal states](../user_guide/initial_terminal_states.md) for the alternatives.

## Hyperparameters

The PPO hyperparameters are those of the [PPO page](ppo.md#hyperparameters). The tracking example differs from
the plain RL example in a few places:

| Key | Value | Why |
|---|---|---|
| `num_steps` | 200 | Longer rollouts per update. |
| `total_timesteps` | 3e8 | Tracking a whole-body motion takes longer than learning to walk on four legs. |
| `init_std` | 0.05 | A small exploration noise; the dense reward leaves little need for exploration. |
| `learnable_std` | false | The standard deviation stays fixed. |
| `control_type` | `DefaultControl` | Direct torque control. The file also contains a commented PD alternative under `control_config`, see [Configuring experiments](configuration.md#the-environment). |

## Validation during training

A high return says that a policy survives, not how well it tracks. The example therefore runs validation
rollouts at regular intervals inside the training function and compares them with the reference using
trajectory similarity measures:

```{literalinclude} ../../examples/training_examples/jax_rl_mimic/conf.yaml
:language: yaml
:start-at: "  validation:"
:end-at: "      - pelvis_rotation"
```

With these settings, validation runs ten times over the course of training. Each time, 100 environments are rolled out for 100
steps, and the Euclidean distance, dynamic time warping and discrete Fréchet distance to the reference are
computed for the joint positions and velocities and for the relative site positions, orientations and
velocities. The results are logged to Weights & Biases under `Validation Measures/`. The script creates the
{class}`~loco_mujoco.utils.metrics.MetricsHandler` and hands it to the algorithm:

```{literalinclude} ../../examples/training_examples/jax_rl_mimic/experiment.py
:language: python
:start-at: mh = MetricsHandler
:end-at: train_fn = PPOJax.build_train_fn
:dedent: 8
```

[Evaluation and trajectory metrics](evaluation.md) explains the measures and every option of the
`validation` block.

## Training in chunks and resuming

`experiment_resume.py` with `conf_resume.yaml` trains the same task in chunks. The training function is built
for `timesteps_per_chunk` steps and called repeatedly until `total_timesteps` is reached. Each call receives
the agent state returned by the previous one:

```{literalinclude} ../../examples/training_examples/jax_rl_mimic/experiment_resume.py
:language: python
:start-at: for chunk in range(n_chunks)
:end-at: agent_state = out["agent_state"]
:dedent: 8
```

The agent state carries the network parameters, the optimizer state, the reward normalization statistics and
the state of all environments, so the boundary between two chunks is seamless. Between chunks, control is
back in Python: metrics are logged while training is still running, and this is the place to save
intermediate checkpoints, change the trajectory, or stop early.

```bash
python experiment_resume.py
python experiment_resume.py experiment.total_timesteps=3e8 experiment.timesteps_per_chunk=1e7
```

In `conf_resume.yaml`, `validation.num` is the number of validations *per chunk*. To continue from a file
instead, load it with `PPOJax.load_agent` and pass the loaded state to the training function, as described in
[Checkpoints and resuming](configuration.md#checkpoints-and-resuming).
