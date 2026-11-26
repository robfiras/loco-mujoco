# Policy distillation with DAgger

Distillation trains a new policy, the *student*, to behave like an existing one, the *teacher*. No reward is
involved: the student learns by supervised regression onto the teacher's actions.
{class}`~loco_mujoco.algorithms.experimental.vanilla_dagger_jax.VanillaDaggerJax` implements this with DAgger
(Dataset Aggregation, [Ross et al., 2011](https://arxiv.org/abs/1011.0686)).

```{note}
The algorithm lives in {mod}`loco_mujoco.algorithms.experimental`. It is newer and less tested than the core
algorithms.
```

## Use cases

- **Many specialists into one generalist.** Train one tracking policy per clip, where each is easy to
  optimise, and distill them into a single policy that performs all clips.
- **A privileged teacher into a deployable student.** Train the teacher with observations that only exist in
  simulation, such as noise-free states or randomized physical parameters, and distill it into a student that
  sees only what the robot's sensors provide.
- **A smaller network.** Compress a large teacher into a network that meets the timing budget of the robot's
  onboard computer.

## How it works

Plain behaviour cloning records the teacher and fits the student to the recording. It fails when the student
makes a small error and reaches a state the teacher never visited, because the data says nothing about what
to do there. DAgger closes that gap by letting the student act during data collection and asking the teacher
what it *would have done* in the states the student reaches.

1. **Rollout with a mixture of both policies.** Each environment is controlled either by the student or by
   the teacher. The choice is made independently per environment: with probability `teacher_beta` the teacher
   acts, otherwise the student, and the choice is kept for a random number of steps between `sticky_min` and
   `sticky_max` before it is drawn again. Letting the teacher act part of the time keeps episodes alive
   early on, when the student still falls quickly.
2. **Label with the teacher.** Whoever acted, the visited observation is stored together with the *teacher's*
   action for it, and the teacher's value estimate.
3. **Supervised update.** The student is a Gaussian policy; it minimises the negative log-likelihood of the
   stored teacher actions on minibatches from a replay buffer,

   $$
   L = -\,\mathbb{E}_{(s,\, a^\text{T}) \sim \mathcal{B}}\big[\log \pi_\text{S}(a^\text{T} \mid s)\big]
       + c\; \mathbb{E}\big[(V_\text{S}(s) - V_\text{T}(s))^2\big].
   $$

   The second term, enabled with `use_critic_learning`, also distills the teacher's value function into the
   student's critic, with weight $c$ = `critic_loss_coef`, so that the student ends up with a value function
   as well as a policy.

Data is kept in two buffers. A circular replay buffer holds the most recent transitions. A second, long-term
buffer keeps a uniform sample of everything seen so far by reservoir sampling, so that early data, for
instance from a teacher that is no longer active, is not forgotten. When the long-term buffer is enabled,
each minibatch is drawn half from each.

Teacher and student are both {class}`~loco_mujoco.algorithms.common.networks.ActorCritic` networks. The
teacher is frozen. Its parameters are part of the agent state and not of the compiled function, so the
teacher can be exchanged between calls of the training function without recompiling.

## Single teacher

`examples/training_examples/jax_distill/experiment.py` distills one PPO teacher. Train the teacher first, for
example with the [DeepMimic example](deepmimic.md), then pass its checkpoint:

```bash
cd examples/training_examples/jax_distill
python experiment.py teacher_ckpt=/abs/path/to/PPOJax_saved.pkl
```

`teacher_ckpt` is mandatory. The environment in `conf.yaml` is the one of the DeepMimic example, a Unitree G1
on the LAFAN1 clip `walk1_subject5`; it has to match the environment the teacher was trained in, since the
teacher is evaluated on its observations. The script trains in chunks of `timesteps_per_chunk` steps, logs
after each chunk to the Weights & Biases project `dagger-distill`, saves `VanillaDaggerJax_saved.pkl` to the
run directory, and records a video of the student.

The teacher is loaded with `PPOJax.load_agent`, and its parameters are handed to the agent state:

```{literalinclude} ../../examples/training_examples/jax_distill/experiment.py
:language: python
:start-at: agent_state = VanillaDaggerJax.init_agent_state
:end-at: train_fn = jax.jit
:dedent: 8
```

Any agent that stores its network as an `ActorCritic` in `train_state` can serve as the teacher in the same
way.

```{important}
The block `experiment.teacher` must describe the architecture the teacher was trained with (`hidden_layers`,
`activation`, `init_std`, `learnable_std`), otherwise its parameters do not fit the network. The values in
`conf.yaml` match the DeepMimic example. The student's architecture under `experiment.student` is independent
and can be smaller.
```

## Several teachers and trajectories

`experiment_traj_swap.py` distills several teachers, each responsible for certain motions, into one student.
Its configuration `conf_traj_swap.yaml` contains a list of pairs of a teacher checkpoint and a trajectory:

```yaml
traj_swap:
  dataset_type: mocap
  pairs:
    - expert_ckpt: /path/to/expert_walk/PPOJax_saved.pkl
      traj: walk
    - expert_ckpt: /path/to/expert_squat/PPOJax_saved.pkl
      traj: squat
    - expert_ckpt: /path/to/expert_squat/PPOJax_saved.pkl
      traj: stepinplace1
```

The trajectory names are tasks of the native dataset, and the example environment is the Unitree H1. The file
in the repository contains placeholder paths; replace them with your own checkpoints before running:

```bash
python experiment_traj_swap.py
```

Training proceeds in chunks. Before each chunk the script picks one pair at random, loads its trajectory
into the environment, puts its teacher into the agent state, and resets the environments:

```{literalinclude} ../../examples/training_examples/jax_distill/experiment_traj_swap.py
:language: python
:start-at: "traj = env.process_trajectory(traj_cache[task])"
:end-at: "out = train_fn(rng, agent_state, traj)"
:dedent: 12
```

The replay buffers are *not* reset. Gradient updates therefore draw on data from all teachers and trajectories
visited so far, which is what lets a single student learn all of them.

Some practical points:

- A teacher that handles several trajectories is listed once per trajectory. Checkpoints are cached by path,
  so repetition costs nothing.
- Each chunk picks uniformly from the list. Listing a pair more often gives it a proportionally larger share
  of the training time. `swap_seed` seeds this choice.
- All teachers must share the architecture declared under `experiment.teacher`.
- With `eval_interval_steps` larger than zero, the student is evaluated on every distinct pair at that
  interval and at the end, deterministically, with `eval_num_envs` environments for `eval_num_steps` steps.
  Return and episode length per pair are logged under `Eval/`.

## Evaluating a student

`eval.py` has two modes:

```bash
# metrics: parallel deterministic rollout, prints mean episode return and length
python eval.py --path /abs/path/to/VanillaDaggerJax_saved.pkl

# viewer and video of one environment
python eval.py --path /abs/path/to/VanillaDaggerJax_saved.pkl --play --deterministic
```

| Flag | Meaning |
|---|---|
| `--path` | The saved agent. Required. |
| `--play` | Render and record step by step instead of computing metrics. |
| `--deterministic` | In play mode, apply the mean action. The metric mode is always deterministic. |
| `--use_teacher` | In play mode, run the teacher stored in the checkpoint instead of the student, as a sanity check. |
| `--use_mujoco` | In play mode, simulate with MuJoCo instead of MJX. |
| `--n_steps` | Number of steps (default 500). |
| `--n_envs` | Number of parallel environments in metric mode. |
| `--seed` | Random seed (default 0). |

## Configuration

| Key | Default in `conf.yaml` | Meaning |
|---|---|---|
| `teacher_ckpt` | required | Path of the teacher checkpoint. Top-level key, single-teacher script only. |
| `student.*`, `teacher.*` | see file | Architecture of each network: `hidden_layers`, `activation`, `init_std`, `learnable_std`. |
| `teacher_beta` | 0.5 | Probability that the teacher acts when an environment draws a new controller. |
| `sticky_min`, `sticky_max` | 5, 50 | Range of the number of steps a controller is kept. |
| `use_critic_learning` | true | Distill the teacher's value function into the student's critic. |
| `critic_loss_coef` | 0.5 | Weight of the value term. |
| `buffer_size` | 1,000,000 | Capacity of the replay buffer of recent transitions. |
| `long_term_buffer_size` | 1,000,000 | Capacity of the reservoir buffer. 0 disables it. |
| `long_term_min_include_prob` | 0.0 | Lower bound on the probability that a new transition enters the reservoir. With 0 the reservoir is a uniform sample of the entire history; a positive value limits it to a recent window of `long_term_buffer_size / long_term_min_include_prob` transitions. |
| `batch_size` | 512 | Minibatch size. |
| `learning_starts` | 4096 | Transitions in the buffer before updates begin. |
| `gradient_steps` | 1 | Updates per vectorized environment step. |
| `lr`, `max_grad_norm`, `weight_decay` | 3e-4, 0.5, 0.0 | AdamW settings. |
| `num_envs` | 2048 | Parallel environments. |
| `total_timesteps`, `timesteps_per_chunk` | 2e7, 1e6 | Total environment steps and steps per call of the training function. |

**Different observations for teacher and student.** `actor_obs_group` and `critic_obs_group` can be set
separately inside `student` and `teacher`, or once under `experiment` for both. This is how a privileged
teacher is distilled into a student with fewer inputs: the environment provides the union of all
observations, the teacher reads its group and the student its own. Groups are defined in the observation
specification, see [Observations](../user_guide/observations.md).

**Memory.** The buffers live on the GPU, and the example scripts compile the training function with
`donate_argnums=(1,)`. This lets JAX reuse the memory of the incoming agent state for the outgoing one instead
of holding two copies of the buffers. A donated agent state must not be used again after the call; the
scripts always continue with the returned one.

**Checkpoints.** The saved file contains the student and the parameters of the teacher that was active last.
The buffers are not saved.
