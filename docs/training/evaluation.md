# Evaluation and trajectory metrics

This page covers the three ways to assess a trained policy: watching it with the `eval.py` scripts, measuring
how closely it follows a reference motion with trajectory similarity measures, and recording videos.

## The evaluation scripts

Most example folders contain an `eval.py`. It needs only the saved agent, because the checkpoint contains the
configuration from which the environment is rebuilt:

```bash
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl               # simulate with MJX
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl --use_mujoco  # simulate with MuJoCo on the CPU
```

The script loads the agent, sets `headless` to false so that the viewer opens, runs the policy and records
a video. The tracking examples also switch the goal to
{class}`~loco_mujoco.core.observations.goals.GoalTrajMimicv2`, which draws the reference motion with the
robot's geometry next to the simulated robot.

| Example | Script | Policy | Notes |
|---|---|---|---|
| `jax_rl`, `jax_rl_mimic`, `jax_gail`, `jax_amp` | `eval.py` | stochastic | 10,000 steps, one environment. |
| `jax_rl_mimic_sac`, `jax_rl_mimic_td3` | `eval.py` | SAC samples from the policy, TD3 adds exploration noise | 10,000 steps, one environment. |
| `jax_rl_mimic_supermimic` | `eval.py` | deterministic | 10,000 steps, one environment. |
| `jax_rl_mimic_supermimic` | `eval_clips.py` | deterministic | Scores every clip of the dataset, see [SuperMimic](supermimic.md#evaluate). |
| `jax_distill` | `eval.py` | deterministic | Prints return and episode length; `--play` renders, see [Distillation](distillation.md#evaluating-a-student). |

**MJX and MuJoCo.** Policies are trained in MJX or MuJoCo Warp. `--use_mujoco` plays the same policy in
MuJoCo, whose contact model and solver differ in details. A policy that only works in the simulator it was
trained in has overfitted to it. Testing in the other simulator is therefore a cheap first check of
robustness, and [domain randomization](../user_guide/domain_randomization.md) during training is the remedy
when it fails.

The scripts are short and meant to be adapted. To change what is evaluated, edit the stored configuration
before the environment is created, for example to play other clips:

```python
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = False
config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name = ["walk1_subject1"]
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
```

## Trajectory metrics

For imitation, return and episode length are weak indicators. They show that the robot stays upright and
collects reward, not how similar its motion is to the reference, and returns cannot be compared between
runs with different rewards. LocoMuJoCo therefore measures the similarity directly. A rollout of the policy
and the reference clip it follows are treated as two time series of the same kinematic quantity, and a
distance between them is computed.

The implementation is {class}`~loco_mujoco.utils.metrics.MetricsHandler`. The distance measures come from the
JAX library [metrx](https://github.com/pompetzki/metrx), so the whole evaluation is compiled and runs on the GPU
inside the training function.

### Distance measures

Let $x_{1:T}$ be the rollout and $y_{1:T}$ the reference, each a sequence of vectors.

| Name in the configuration | Definition | Behaviour |
|---|---|---|
| `EuclideanDistance` | The mean over time of $\lVert x_t - y_t \rVert$. | Compares the two series step by step. A motion that is correct but early or late is penalised. |
| `DynamicTimeWarping` | The sum of the distances along the best monotone alignment of the two series. | Insensitive to differences in timing. Being a sum, it grows with the length of the rollout. |
| `DiscreteFrechetDistance` | The largest distance along the best monotone alignment. | Reports the worst deviation instead of the average one. |

Using the three together separates different kinds of error: a large Euclidean distance with a small DTW value
points to a timing offset, and a large Fréchet distance with a small Euclidean one to a short, large deviation.
All values are distances, so lower is better.

### Quantities

A measure is evaluated on one or more of eleven quantities. The results are stored in a
{class}`~loco_mujoco.utils.metrics.QuantityContainer`, one field per quantity.

| Name in the configuration | Field | Compared quantity |
|---|---|---|
| `JointPosition` | `qpos` | Joint positions. The quaternion of a free joint is converted to a rotation vector. |
| `JointVelocity` | `qvel` | Joint velocities. |
| `BodyPosition` | `xpos` | Positions of bodies in the world frame. |
| `BodyOrientation` | `xrotvec` | Orientations of bodies in the world frame, as rotation vectors. |
| `BodyVelocity` | `cvel` | Six-dimensional velocities of bodies. |
| `SitePosition` | `site_xpos` | Positions of sites in the world frame. |
| `SiteOrientation` | `site_xrotvec` | Orientations of sites in the world frame, as rotation vectors. |
| `SiteVelocity` | `site_xvel` | Velocities of sites. |
| `RelSitePosition` | `site_rpos` | Positions of sites relative to a main site. |
| `RelSiteOrientation` | `site_rrotvec` | Orientations of sites relative to a main site, as rotation vectors. |
| `RelSiteVelocity` | `site_rvel` | Velocities of sites relative to a main site. |

The relative site quantities are the most informative for whole-body imitation. They describe the pose of
the hands, feet and head relative to the torso and are the quantities the
[tracking reward](deepmimic.md#the-reward) is mainly built on. The world-frame quantities also include
drift of the whole robot away from the reference path.

Each measure is computed separately for every element (joint, body or site) and every validation environment
on the time series of that element, and then averaged. The result is one scalar per measure and quantity.
Quantities with different units are not comparable with each other; compare the same quantity across runs.

### Configuration

The validation loop is configured in the `validation` block of `experiment`:

```{literalinclude} ../../examples/training_examples/jax_rl_mimic/conf.yaml
:language: yaml
:start-at: "  validation:"
:end-at: "      - pelvis_rotation"
```

| Key | Meaning |
|---|---|
| `active` | Whether the example script creates a metrics handler at all. |
| `num` | Number of validations during training. They are spaced evenly: one every `num_updates / num` updates. |
| `num_envs` | Number of validation environments, each producing one rollout. |
| `num_steps` | Length of the validation rollouts in steps. |
| `quantities` | List of quantities from the table above. |
| `measures` | List of distance measures. Requires an environment with a trajectory. |
| `rel_site_names` | Sites used for the site quantities. For the relative quantities the first entry is the main site. Default: all sites of the model. |
| `rel_body_names` | Bodies used for the body quantities. Default: all bodies. |
| `rel_joint_names` | Joints used for the joint quantities. Default: all joints. |
| `joints_to_ignore` | Joints excluded from the joint quantities, for example the root joint if only the posture is of interest. |

To disable validation, set `active: false`. Leave `num` at a positive value in that case, since it is still
used to compute the validation interval.

Each validation resets `num_envs` fresh environments, rolls out the current policy for `num_steps` steps with
actions sampled from the policy, and passes the recorded simulation states to the metrics handler. The
validation environments are independent of the training environments.

```{note}
The handler stores the complete simulation state of all validation environments for the length of the
rollout. With large `num_envs` and `num_steps` this needs a noticeable amount of GPU memory.
```

### What is logged

The training function returns the validation results as a
{class}`~loco_mujoco.utils.metrics.ValidationSummary` with one entry per update; entries of updates without a
validation are zero. Its fields are `mean_episode_return`, `mean_episode_length`, and one `QuantityContainer`
for each measure: `euclidean_distance`, `dynamic_time_warping` and `discrete_frechet_distance`. Quantities that
were not requested are empty arrays.

The example scripts walk through this structure and log every non-empty field at the updates where a
validation took place:

| Weights & Biases key | Content |
|---|---|
| `Validation Info/Mean Episode Return` | Mean return of the episodes that ended during the validation rollouts. |
| `Validation Info/Mean Episode Length` | Their mean length. |
| `Validation Measures/<measure>/<field>` | One curve per measure and quantity, for example `Validation Measures/dynamic_time_warping/site_rpos`. |

With the configuration above, that is three measures times five quantities: fifteen curves.

## Calling the metrics from your own code

The metrics handler is created from a configuration with a `validation` block and an environment that has a
trajectory. Passed to `build_train_fn`, it is used during training, as in the examples. Passed to
`build_eval_fn`, it evaluates an existing agent without training:

```python
import jax
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils import MetricsHandler

agent_conf, agent_state = PPOJax.load_agent("outputs/<date>/<time>/PPOJax_saved.pkl")
config = agent_conf.config

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

mh = MetricsHandler(config, env)
eval_fn = jax.jit(PPOJax.build_eval_fn(env, agent_conf, mh=mh))
out = eval_fn(jax.random.PRNGKey(0), agent_state, traj)

print(out["eval_summary"].mean_episode_return)
metrics = out["validation_metrics"]
print(metrics.euclidean_distance.qpos, metrics.dynamic_time_warping.site_rpos)
```

The evaluation function has the signature of the training function. It rolls out
`validation.num_envs` environments for `validation.num_steps` steps and returns a dictionary with
`eval_summary` (episode return and length) and `validation_metrics`. Change the two values in `config` before
building the function to evaluate on more or longer rollouts. `build_eval_fn` is implemented for
{class}`~loco_mujoco.algorithms.ppo_jax.PPOJax` and
{class}`~loco_mujoco.algorithms.experimental.vanilla_dagger_jax.VanillaDaggerJax`.

To compare two arbitrary time series, use the distance measures of metrx directly. The inputs are arrays of
shape `(time, dimension)`:

```python
import jax.numpy as jnp
from metrx import DistanceMeasures

dtw = DistanceMeasures.create_instance("DynamicTimeWarping")
euclidean = DistanceMeasures.create_instance("EuclideanDistance", mean=True)

x = jnp.zeros((50, 3))
y = jnp.ones((50, 3))
print(dtw(x, y), euclidean(x, y))
```

The names of all supported measures and quantities are available as
`MetricsHandler.supported_measures` and `MetricsHandler.supported_quantities`.

## Recording videos

Every algorithm has a `play_policy` method that rolls out an agent and can record it:

```python
PPOJax.play_policy(env, agent_conf, agent_state, n_envs=20, n_steps=200,
                   deterministic=True, record=True, traj=traj)
print(env.video_file_path)
```

| Argument | Meaning |
|---|---|
| `n_envs` | Number of environments simulated in parallel and rendered into the same scene. |
| `n_steps` | Number of steps. Without it the rollout runs until it is interrupted. |
| `deterministic` | Apply the mean action instead of sampling. |
| `record` | Write a video. The file is placed in `LocoMuJoCo_recordings/` and its path is available as `env.video_file_path` afterwards. |
| `train_state_seed` | For agents trained with several seeds (PPO and its relatives): the index of the seed to play. |
| `traj` | The trajectory of an imitation environment. |

Recording does not need a display. With `headless: true` in `env_params`, as in the training configurations,
frames are rendered off-screen; this is how the examples record a video at the end of training on a cluster
and upload it to Weights & Biases. With `headless: false` the viewer window opens as well.

For MuJoCo instead of MJX, `PPOJax` and `GAILJax` provide `play_policy_mujoco`; the other algorithms take
`use_mujoco=True` in `play_policy`. Only one environment can be played in that mode.

To convert a recording to a GIF, for example for a README, use
{func}`~loco_mujoco.utils.video.video2gif`. Camera settings and the other rendering options are described in
[Rendering](../user_guide/rendering.md).
