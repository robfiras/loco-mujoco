# Configuring experiments

Every training example is configured by a YAML file that is read with [Hydra](https://hydra.cc). The file
describes the complete experiment: which robot and dataset, which reward and observations, and all
hyperparameters of the algorithm. This page explains the structure of these files and how to run, vary and
monitor experiments with them.

## Anatomy of a configuration

The scripts load `conf.yaml` from their own folder:

```python
@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    ...
```

A configuration has up to four top-level groups.

| Group | Contents |
|---|---|
| `defaults` | Hydra settings. The examples select Hydra's default logging and the basic (sequential) launcher. |
| `wandb` | `project`: the Weights & Biases project the run is logged to. |
| `experiment` | The experiment itself. Its entries `task_factory` and `env_params` define the environment; all remaining entries are read by the algorithm. |
| `control_config` | Present in the tracking examples. A named collection of controller settings that `env_params` refers to by interpolation, see below. |

Abridged, the DeepMimic configuration looks like this:

```yaml
wandb:
  project: "deepmimic"

experiment:
  task_factory:                 # how the environment is created
    name: ImitationFactory
    params:
      lafan1_dataset_conf:
        dataset_name:
          - walk1_subject5
  env_params:                   # keyword arguments of the environment
    env_name: MjxUnitreeG1
    headless: True
    horizon: 1000
    goal_type: GoalTrajMimic
    goal_params:
      visualize_goal: true
    reward_type: MimicReward
    reward_params:
      qpos_w_sum: 0.4
      ...
  hidden_layers: [512, 256]     # everything else: the algorithm
  lr: 1e-4
  num_envs: 2048
  ...
  validation:                   # periodic validation rollouts
    active: true
    ...
```

### The environment

The scripts create the environment with two lines:

```python
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
```

`task_factory.name` is {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` for a task without motion
data or {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` for one with.
`task_factory.params` are the factory's own arguments: for `ImitationFactory` the dataset configurations
`default_dataset_conf`, `lafan1_dataset_conf`, `amass_dataset_conf` and `custom_dataset_conf`, described in
[Task factories](../user_guide/task_factories.md).

`env_params` are passed on to the environment. `env_name` selects the robot; the prefix `Mjx` selects the GPU
version that the JAX algorithms require. Every pluggable module of the environment is set with a pair of keys:
`<module>_type` names a registered class and `<module>_params` holds its arguments.

| Keys | Module | Guide |
|---|---|---|
| `reward_type`, `reward_params` | Reward function | [Rewards](../user_guide/rewards.md) |
| `goal_type`, `goal_params` | Goal, which is appended to the observation | [Observations](../user_guide/observations.md) |
| `control_type`, `control_params` | Control function, for example torque or PD position control | [Control functions](../user_guide/control_functions.md) |
| `domain_randomization_type`, `domain_randomization_params` | Domain randomizer | [Domain randomization](../user_guide/domain_randomization.md) |
| `terrain_type`, `terrain_params` | Terrain | [Terrain](../user_guide/terrain.md) |
| `init_state_type`, `init_state_params` | Initial state handler | [Initial and terminal states](../user_guide/initial_terminal_states.md) |
| `terminal_state_type`, `terminal_state_params` | Terminal state handler | [Initial and terminal states](../user_guide/initial_terminal_states.md) |
| `th_params` | Trajectory handler: `name` and its arguments | [Trajectories](../user_guide/trajectories.md) |

Other commonly used entries of `env_params` are `horizon` (the episode length in steps), `headless` (no
viewer window; keep it `true` for training), and `use_mjwarp` (simulate with MuJoCo Warp instead of MJX, see
[Simulation backends](../user_guide/backends.md)). A module that is not mentioned keeps the default of the
environment. Classes you [write and register yourself](../user_guide/custom_modules.md) can be named in the
same way, provided the module that defines them is imported before the environment is created.

For example, adding domain randomization to an experiment takes one block:

```yaml
  env_params:
    ...
    domain_randomization_type: DefaultRandomizer
    domain_randomization_params:
      randomize_geom_friction_tangential: true
      add_joint_pos_noise: true
```

Observations are configured in two places. Every robot has a default observation space, which
`env_params.observation_spec` replaces: a list of entries, each with the `type` of a registered observation
and the arguments of that class, such as `obs_name`, `xml_name` and `group`.

```yaml
  env_params:
    ...
    observation_spec:
      - {type: FreeJointPosNoXY, obs_name: q_root, xml_name: root, group: policy}
      - {type: JointPos, obs_name: q_left_knee_joint, xml_name: left_knee_joint, group: policy}
      ...
```

The observation types and groups are described in [Observations](../user_guide/observations.md). Which
*groups* the networks read is then a setting of the algorithm: `actor_obs_group` and `critic_obs_group`, see
[PPO](ppo.md#network-and-policy-options).

### Interpolation

Hydra can refer from one entry to another with `${...}`. The tracking examples use this to keep alternative
controller settings side by side:

```yaml
experiment:
  env_params:
    control_type: ${control_config.torque_control.control_type}
    control_params: ${control_config.torque_control.control_params}

control_config:
  position_control:
    control_type: PDControl
    control_params:
      p_gain: [...]
      d_gain: [...]
  torque_control:
    control_type: DefaultControl
    control_params:
```

Changing `torque_control` to `position_control` in the two references switches the controller. The gains listed
under `position_control` are specific to one robot; adapt them when you change `env_name`.

### Derived values

`init_agent_conf` writes values it computes into `config.experiment`, for PPO `num_updates`,
`minibatch_size` and `validation_interval`. The configuration stored in a checkpoint therefore contains a few
keys more than the YAML file.

## Command-line overrides

Any entry can be changed on the command line without editing the file. The path to an entry is written with
dots:

```bash
python experiment.py experiment.lr=3e-4 experiment.num_envs=4096 wandb.project=my_project
```

| Syntax | Effect |
|---|---|
| `a.b=value` | Change an existing entry. |
| `+a.b=value` | Add an entry that is not in the file. |
| `~a.b` | Remove an entry. |
| `'a.b=[x,y]'` | Set a list. Quote the argument so that the shell does not interpret the brackets. |
| `'+a.b={k: v}'` | Add a dictionary. |

Some examples for the DeepMimic configuration:

```bash
# other clips
python experiment.py 'experiment.task_factory.params.lafan1_dataset_conf.dataset_name=[walk1_subject1,dance2_subject4]'

# another robot and the native dataset instead of LAFAN1
python experiment.py experiment.env_params.env_name=MjxUnitreeH1 \
    '~experiment.task_factory.params.lafan1_dataset_conf' \
    '+experiment.task_factory.params.default_dataset_conf={task: walk}'

# add a module that the file does not mention
python experiment.py +experiment.env_params.domain_randomization_type=DefaultRandomizer

# a different network
python experiment.py 'experiment.hidden_layers=[1024,512,256]' experiment.activation=elu
```

To keep several variants of a configuration, copy `conf.yaml` under another name in the same folder and
select it with `--config-name`:

```bash
python experiment.py --config-name conf_h1
```

## Seeds and sweeps

**Seeds.** The on-policy examples (PPO, DeepMimic, GAIL, AMP, S2PG, BPTT-PPO, history-aware PPO) run
`experiment.n_seeds` seeds in parallel on one GPU with `jax.vmap`. Seed `i` is `jax.random.PRNGKey(i)` for
`i = 1 ... n_seeds`. The logged curves are the mean over the seeds, and the saved agent contains all of them;
`play_policy` takes `train_state_seed` to choose one. Memory grows linearly with the number of seeds. The
SuperMimic, SAC, TD3 and distillation examples train a single seed.

**Sweeps.** Hydra's multirun mode runs the script once for every combination of comma-separated values:

```bash
python experiment.py --multirun experiment.lr=1e-4,3e-4 experiment.clip_eps=0.1,0.2
```

This starts four runs, one after the other, each with its own Weights & Biases run. The same sweep can be
written into the configuration; the examples contain a commented template:

```yaml
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      experiment.lr: 1e-4, 3e-4
      experiment.clip_eps: 0.1, 0.2
```

To distribute the runs of a sweep over a cluster, replace the `basic` launcher in `defaults` by one of
[Hydra's launcher plugins](https://hydra.cc/docs/plugins/submitit_launcher/).

The on-policy examples also log a scalar `Metric for Sweep` as an objective for a hyperparameter sweep: the sum
of the Euclidean errors of the relative site positions, orientations and velocities. It is logged only if
`validation.quantities` contains all three (`RelSitePosition`, `RelSiteOrientation`, `RelSiteVelocity`) and
`validation.measures` contains `EuclideanDistance`.

## Output directories

A run writes to three places, all relative to the directory the script is started from:

| Location | Contents |
|---|---|
| `outputs/<date>/<time>/` | The run directory created by Hydra. It contains the saved agent `<ClassName>_saved.pkl`, the log file `experiment.log`, and the folder `.hydra/` with `config.yaml` (the complete configuration of the run), `overrides.yaml` (the command-line overrides) and `hydra.yaml`. |
| `multirun/<date>/<time>/<n>/` | The same, for run number `n` of a sweep. |
| `LocoMuJoCo_recordings/` | The videos recorded at the end of training and by the evaluation scripts. |
| `wandb/` | The local files of Weights & Biases. |

The run directory can be set explicitly with `hydra.run.dir=/path/to/run`. The path of the saved agent is
also printed at the end of training and added to the Weights & Biases run configuration as
`agent_save_path`.

## Weights and Biases

The scripts call `wandb.login()` and `wandb.init(project=config.wandb.project, config=...)`. The complete
configuration is attached to the run, so runs can be filtered and grouped by any hyperparameter. Three
kinds of data are logged:

- `Training/...`: the fields of the algorithm's summary metrics, such as episode return, episode length and
  losses, with the number of environment steps as the x-axis;
- `Validation Info/...` and `Validation Measures/...`: the results of the [validation loop](evaluation.md), if
  it is active;
- `Agent Video`: the recording of the trained policy.

Because training is one compiled function call, the curves are uploaded after it returns, see
[how training works](index.md#how-training-works).

Weights & Biases is controlled through its own environment variables. The most useful ones here:

| Setting | Effect |
|---|---|
| `wandb.project=my_project` (override) | Log to another project. |
| `WANDB_ENTITY=my_team` | Log to a team instead of your personal account. |
| `WANDB_MODE=offline` | Write the run to the local `wandb/` folder without contacting the server and without needing an account. Upload it later with `wandb sync <folder>`. |
| `WANDB_MODE=disabled` | Turn logging off entirely. The script runs unchanged and nothing is written. |

```bash
WANDB_MODE=disabled python experiment.py experiment.total_timesteps=1e7   # a quick test without logging
```

## Checkpoints and resuming

**What is saved.** `save_agent(path, agent_conf, agent_state)` writes one pickle file,
`<ClassName>_saved.pkl`, into the directory `path`. It contains the resolved configuration, the network
definitions, the parameters, the optimizer state and the normalization statistics. `load_agent(file)` returns
the pair `(agent_conf, agent_state)`; nothing else is needed to play or continue training, because the
environment can be rebuilt from the stored configuration. This is what the evaluation scripts do. The state of
the environments and, for the off-policy algorithms and distillation, the replay buffers are not saved.

**When it is saved.** The example scripts save once, after training. A run that is interrupted leaves no
checkpoint.

**Training in chunks.** To obtain intermediate checkpoints, or to see metrics while training runs, call the
training function several times for a part of the total steps each. The agent state returned by one call
is the input of the next. `jax_rl_mimic/experiment_resume.py`, `jax_rl_mimic_sac/experiment_resume.py` and the
distillation scripts work this way, with the chunk length given by `experiment.timesteps_per_chunk`:

```python
train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf, mh=mh))   # built for timesteps_per_chunk steps

for chunk in range(n_chunks):
    out = train_fn(rng, agent_state, traj)
    agent_state = out["agent_state"]
    # log out["training_metrics"], save a checkpoint, ...
```

The function is compiled once and reused for all chunks. `save_agent` always uses the same file name, so pass a
different directory for each checkpoint you want to keep.

**Resuming from a file.** The same mechanism continues a finished or saved run:

```python
agent_conf, agent_state = PPOJax.load_agent("outputs/<date>/<time>/PPOJax_saved.pkl")
config = agent_conf.config

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))
out = train_fn(jax.random.PRNGKey(0), agent_state, traj)    # another total_timesteps steps
```

Parameters, optimizer state and normalization statistics continue from the file; the environments start from
a reset. Two algorithm-specific points: GAIL and AMP need their demonstrations attached again with
`add_expert_dataset` before training continues, and an agent trained with several seeds has to be resumed
with the training function vectorized over the seeds, as in the original script.

## GPU memory and the number of environments

The number of parallel environments is the main lever for both speed and memory.

- **On-policy algorithms** hold a rollout buffer of `num_envs * num_steps` transitions, multiplied by
  `n_seeds`. If the GPU runs out of memory, reduce `num_envs`, `num_steps` or `n_seeds`. Remember that
  `num_envs * num_steps` must stay divisible by `num_minibatches` (for BPTT-PPO, `num_envs` by
  `num_minibatches`).
- **Changing `num_envs` changes the optimisation.** With fewer environments each update sees less data.
  Increasing `num_steps` by the same factor keeps the batch size constant.
- **The validation loop** stores the full simulation state of `validation.num_envs` environments for
  `validation.num_steps` steps. Keep both moderate; the examples use 100 and 100.
- **Off-policy algorithms and distillation** hold replay buffers of `buffer_size` transitions on the GPU. The
  size of a transition is dominated by the observation dimension.
- **SuperMimic** differentiates through imagined rollouts. `remat: true` trades compute for memory there, and
  `imag_batch_size` and `imag_horizon` scale the cost of a policy update.
- **Contact-rich scenes with MuJoCo Warp** need the contact and constraint buffers sized through `nconmax` and
  `njmax` in `env_params`, see [Simulation backends](../user_guide/backends.md).

By default JAX reserves most of the GPU memory when it starts. On a shared GPU, set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to limit the share, or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to allocate
on demand.

```{tip}
Before a long run, check a new configuration with a short one, for example
`WANDB_MODE=disabled python experiment.py experiment.total_timesteps=1e7`. It goes through compilation, a few
updates, saving and the final recording, which is where configuration mistakes show up.
```
