# Train a tracking policy

<ul class="lmj-steps">
  <li><a href="installation.html">1 · Installation</a></li>
  <li><a href="first_environment.html">2 · First environment</a></li>
  <li><a href="replay_motion.html">3 · Replay a motion</a></li>
  <li class="current">4 · Train a policy</li>
  <li><a href="next_steps.html">5 · Evaluate and customize</a></li>
</ul>

On this page you train a policy that makes the simulated Unitree G1 follow a walking clip. The policy is
trained with PPO and a DeepMimic-style tracking reward on 2048 parallel MJX environments.

```{image} ../../imgs/unitree_h1_walk_anydir.gif
:alt: A Unitree H1 walking in all directions with a policy trained by this example
:width: 70%
:align: center
```

<p class="lmj-caption">A policy trained with this example on the Unitree H1, on a dataset of walking in all
directions.</p>

## Before you start

The example is part of the repository, not of the PyPI package. If you installed LocoMuJoCo from PyPI, clone
the repository to get it:

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
```

You need a GPU with the [CUDA build of JAX](installation.md#add-gpu-support) and Weights & Biases for logging:

```bash
pip install wandb
```

The example logs in to Weights & Biases when it starts. To run without an account, set `WANDB_MODE=disabled`
in your shell.

## Run the example

```bash
cd examples/training_examples/jax_rl_mimic
python experiment.py
```

The run trains for 300 million environment steps. On a single modern GPU this takes roughly half an hour
to an hour; the Unitree H1 version of this example took 36 minutes on an RTX 3080 Ti. The first minutes are
spent compiling, during which nothing is printed.

When training finishes, the script

- saves the agent as `PPOJax_saved.pkl` in a time-stamped folder under `outputs/`,
- renders the trained policy and stores a video in `LocoMuJoCo_recordings/`,
- uploads the training curves, the validation metrics and the video to Weights & Biases.

To try the pipeline quickly before committing to a full run, shorten it from the command line:

```bash
python experiment.py experiment.total_timesteps=2e7 experiment.num_envs=1024
```

The result will not walk yet, but you will see every stage run.

## What the example does

The whole experiment is described by `conf.yaml` in the example folder. Its main parts are:

```yaml
experiment:
  task_factory:
    name: ImitationFactory
    params:
      lafan1_dataset_conf:
        dataset_name:
          - walk1_subject5          # the clip to imitate
  env_params:
    env_name: MjxUnitreeG1          # the MJX version of the Unitree G1
    goal_type: GoalTrajMimic        # the policy observes the reference motion
    reward_type: MimicReward        # DeepMimic-style tracking reward
    reward_params:
      qpos_w_sum: 0.4               # weight of the joint position term
      qvel_w_sum: 0.2               # joint velocities
      rpos_w_sum: 0.5               # relative site positions
      rquat_w_sum: 0.3              # relative site orientations
      rvel_w_sum: 0.1               # relative site velocities
  num_envs: 2048
  total_timesteps: 300e6
```

Three choices turn a plain reinforcement learning setup into motion tracking:

The dataset
: `ImitationFactory` loads the clip, as on the [previous page](replay_motion.md). Episodes start at random
  frames of the clip and end when the robot deviates too far from it.

The goal
: {class}`~loco_mujoco.core.observations.goals.GoalTrajMimic` adds the current reference pose to the observation, so
  the policy knows what it should be doing at each moment.

The reward
: {class}`~loco_mujoco.core.reward.trajectory_based.MimicReward` rewards the policy for matching the joint
  positions and velocities of the reference and the relative positions, orientations and velocities of a few
  key sites on the hands, the feet and the upper body.

The training script itself is short. It creates the environment from the configuration, builds a training
function and compiles environment and learner into a single program:

```python
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

agent_conf = PPOJax.init_agent_conf(env, config)
agent_state = PPOJax.init_agent_state(env, agent_conf, rng)

train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))
out = train_fn(rng, agent_state, traj)

PPOJax.save_agent(result_dir, agent_conf, out["agent_state"])
```

Because simulation, rollout collection and gradient updates all run inside one compiled function on the
GPU, no data moves between the CPU and the GPU during training.

## Watch the training

During training the policy is evaluated ten times on a separate set of environments. Besides the return and
the episode length, the evaluation compares the motion of the policy with the reference motion using three
distance measures, Euclidean distance, dynamic time warping and the discrete Fréchet distance, computed on
joint positions, joint velocities and site poses. All of these appear in the Weights & Biases run under
*Validation Measures*. A falling distance means the robot moves more like the reference.

## Try other motions

Change the clip or add more of them in `conf.yaml`, or override the list from the command line:

```bash
python experiment.py "experiment.task_factory.params.lafan1_dataset_conf.dataset_name=[walk1_subject1,run1_subject2]"
```

A single policy can track several clips. More clips and more dynamic motions need more training steps.

## What you learned

- A training example is a `conf.yaml` plus a short script; everything in the configuration can be
  overridden from the command line.
- Motion tracking is PPO with a reference-motion goal and a tracking reward.
- The trained agent is a single pickle file that contains the policy and its configuration.

Next: [evaluate and customize](next_steps.md).
