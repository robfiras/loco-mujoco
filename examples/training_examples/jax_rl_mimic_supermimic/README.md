## SuperMimic Imitation Learning Example

This example trains a motion tracking policy for the Unitree G1 with **SuperMimic**
([paper](https://arxiv.org/abs/2610.28480)) on a LAFAN1 walking clip (`walk1_subject5`).

SuperMimic fits a world model on the transitions it collects and improves the policy by backpropagating
through imagined rollouts of that model, with a TD(λ) critic for the tail of the rollout. The observations,
rewards and terminations of an imagined step are the environment's own JAX functions, so the same example works
with other robots, goals and rewards. Here the goal is `GoalTrajMimic` and the reward is `MimicReward`, as in the
[DeepMimic example](../jax_rl_mimic). See `loco_mujoco/algorithms/supermimic_jax.py` for the algorithm.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the agent on the Unitree G1 with 4096 parallel environments for 1.31 billion steps.
- Save the trained agent (as `SuperMimicJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video and the training metrics to Weights & Biases (WandB).

All settings are in `conf.yaml` and can be overridden on the command line, for example to train on other clips or
for fewer steps:

```bash
python experiment.py 'experiment.task_factory.params.lafan1_dataset_conf.dataset_name=[walk1_subject1,dance2_subject4]' experiment.total_timesteps=3e8
```

A few settings worth knowing:

- `env_params.use_mjwarp`: with `true` the environment steps run on MjWarp, and imagination uses a JAX copy of
  the model.
- `env_params.domain_randomization_type`: with a randomizer, its per-episode state is given to the world model
  and the critic as context.
- `actor_obs_group`, `critic_obs_group`, `termination_obs_group`: observation groups for the actor, the critic
  and the termination model. By default all three get the full observation.
- `imag_horizon`, `imag_batch_size`, `updates_per_env_step`: length and number of the imagined rollouts and the
  number of updates per environment step.

> **Note:** SuperMimic's training function needs `jax<=0.10.1,!=0.10.0` to compile on the GPU, which is what
> LocoMuJoCo installs.

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```

`eval_clips.py` scores a checkpoint on every clip of its dataset, without noise or domain randomization. Each clip
is played once from its first frame; the script prints the normalized return per clip and how many clips
terminated early. It reads `config.yaml` from the run directory, which Hydra stores under `.hydra/`:

```bash
cp path/to/run/.hydra/config.yaml path/to/run/
```

```bash
python eval_clips.py path/to/run SuperMimicJax_saved.pkl
```

---

### 🤖 On a Real Robot

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) has deployable SuperMimic training setups for the
Unitree G1 and the AgiBot X2 Ultra (`training/<robot>/train_supermimic.py`) and the tools to take the
checkpoint to the robot.
