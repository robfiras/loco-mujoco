# Vanilla DAgger distillation (jax_distill)

Distill a pretrained teacher policy into a student via **Vanilla DAgger**.
Rollouts use a sticky per-env mixture of student/teacher actions; every
observed state is labeled with the *teacher's* action and stored in a replay
buffer. The student is trained by minimizing the NLL of its Gaussian policy
at the teacher action.

See `loco_mujoco/algorithms/experimental/vanilla_dagger_jax.py` for the
algorithm.

---

## Files

- `experiment.py` — single-teacher distillation with Hydra + wandb logging.
- `conf.yaml` — its config (override `teacher_ckpt=/path/to/PPOJax_saved.pkl`).
- `experiment_traj_swap.py` — **headline example**: multiple (trajectory,
  teacher) pairs. At each training chunk the script picks one at random and
  swaps both into the agent state. The replay buffer and rollout mixture
  state **persist across swaps**, so data collected under an earlier teacher
  stays useful.
- `conf_traj_swap.yaml` — list of tasks + their pretrained teacher
  checkpoints.
- `eval.py` — load a saved student and play it (or `--use_teacher` to
  sanity-check the teacher).

## Minimal usage

Train a PPO teacher first (e.g. via `../jax_rl_mimic`), then:

```bash
python experiment.py teacher_ckpt=/abs/path/to/PPOJax_saved.pkl
```

For the traj-swap setup, edit `conf_traj_swap.yaml` to list your (task,
teacher_ckpt) pairs, then:

```bash
python experiment_traj_swap.py
```

Play the distilled student:

```bash
python eval.py --path /abs/path/to/VanillaDaggerJax_saved.pkl --deterministic
```

## Architecture constraints

The teacher checkpoint must match the architecture declared under
`experiment.teacher` in the config (same `hidden_layers`, `activation`,
`init_std`, `learnable_std`). If you load a teacher whose checkpoint was
produced with different hyperparameters, Flax will fail to deserialize.

Student architecture is independent — it's declared under
`experiment.student` and can be smaller/simpler than the teacher (typical
distillation use case).

## Swap semantics

All three swappable entities live on the JAX pytree:

- **traj**: passed as an arg to `train_fn(rng, agent_state, traj)`. Call
  `env.process_trajectory(new_traj)` between chunks.
- **teacher**:
  ```python
  agent_state.replace(teacher_params=new_params,
                      teacher_run_stats=new_run_stats)
  ```
- **student**:
  ```python
  agent_state.replace(student_train_state=new_ts)
  ```

Between chunks, also null the env carry so the env resets cleanly for the
new trajectory indices:

```python
agent_state = agent_state.replace(env_state=None, last_obs=None)
```

The replay buffer is **preserved** across this reset — that's the core
reason for the agent-state layout.
