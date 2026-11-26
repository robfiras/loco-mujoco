# Changelog

All notable changes to LocoMuJoCo are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers follow the pattern
`MAJOR.MINOR.PATCH`; breaking changes are listed under their own heading in each release.

Every version below is a git tag and a
[GitHub release](https://github.com/robfiras/loco-mujoco/releases). The dates are the dates of the tags.

## [Unreleased]

## [1.2.1] - 2026-11-23

### Added

- New documentation site: a getting-started track, a user guide with one page per environment
  component, a page per training algorithm, a generated environment gallery, a sim-to-real page, an API
  reference generated from the docstrings, and developer documentation (architecture, testing, writing
  documentation, release process, comparison to related software).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md` and `CITATION.cff`, and templates for issues
  and pull requests.
- Continuous integration builds the documentation with warnings treated as errors and checks docstring
  coverage. New optional dependency group `docs`.
- Continuous integration builds the wheel and the source distribution, installs the wheel into a clean
  environment on Python 3.10, 3.11, 3.12 and 3.13, and runs a smoke test with it; this job also runs weekly.
  The package lists the supported Python versions in its classifiers.
- The simulation backends page reports throughput and peak GPU memory against mjlab and MuJoCo Playground,
  with both frameworks running the same task on the same model.

### Changed

- The README, the documentation and `CITATION.cff` cite the paper on the current version in addition to
  the original 2023 paper.
- The documentation moved from reStructuredText and the Read the Docs theme to MyST Markdown and the
  PyData theme. The old page addresses redirect to the new ones.
- Docstrings of the public API were completed; coverage is enforced with `interrogate` (threshold in
  `pyproject.toml`). No behaviour changed.
- `README.md` and the package READMEs were shortened and now point to the documentation site.
- Linux is stated as the supported operating system, in the installation guide and in the package
  classifiers (`Operating System :: POSIX :: Linux` instead of `OS Independent`). Continuous integration
  runs on Linux only; macOS and Windows are not tested.
- The license is declared as the SPDX expression `MIT` with `license-files`, as setuptools 77 and newer
  expect; the deprecated license table and license classifier are removed.

### Removed

- `examples/training_examples/jax_rl_mimic_s2pg4`. The example imported `S2PG4Jax`, an algorithm that is not
  part of LocoMuJoCo, and could not be run.

### Fixed

- A wheel or source distribution built from a working copy contained the bytecode caches (`__pycache__`) of
  `loco_mujoco/smpl`. `setup.py` and `MANIFEST.in` now exclude them, and `MANIFEST.in` no longer includes the
  non-existent `requirements.txt`.
- `examples/tutorials/11_creating_custom_modules.py` failed at `env.reset()` because its custom modules did
  not accept the `traj` argument introduced in 1.2.0.
- Stale statements in the READMEs of the training examples (robot, algorithm and file names).
- `anneal_lr: true` failed in `PPOJax`, `GAILJax` and `AMPJax` with a `ConfigAttributeError`, because the
  optimizer read the learning rate from `config.lr` instead of `config.experiment.lr`.
- `validation.num: 0` divided by zero in `init_agent_conf` of the PPO-based algorithms. It now disables the
  validation rollouts, as the example configurations say. A `validation.num` larger than the number of updates
  no longer leads to a validation interval of zero.
- `play_policy(use_mujoco=True, wrap_env=True)` of `PPOJax` and `GAILJax` raised an `AttributeError`
  (`agent_conf.experiment` instead of `agent_conf.config.experiment`).
- `play_policy(deterministic=True)` of the PPO-based algorithms no longer overwrites `log_std` in the
  parameters of the agent state that is passed in.
- `BestTrainStates.create` raised a `TypeError` because it did not set the field `step`.
- `DefaultRandomizer` required every configuration key and raised a `KeyError` at the first reset if one was
  missing. Switches that are not given are now off, and a switch that is on without its range or scale raises a
  `ValueError` at construction. Complete configurations behave as before. The tutorial
  `08_domain_randomization.py` no longer sets the two `joint_stiffness` keys, which were never read.
- The errors for a trajectory with more samples or clips than the trajectory handler holds (100,000 samples and
  100 clips by default) now say how to raise the limits: `th_params=dict(max_n_samples=..., max_n_trajs=...)`.
- `video_file_path` of `MujocoViewer` and `ViserViewer` raised an `AttributeError`; `VideoRecorder` has a new
  property `file_path` that both use.
- The "Ran out of geoms" error of `MujocoViewer` failed with an `AttributeError` while building its message,
  and the capacity checks were off by one.
- `MjvGeom` declared the field `type` twice, and `TargetVelocityGoalReward` called the constructor of its base
  class twice.
- `MyoSkeleton` ended the Python process with `exit()` when its model had not been downloaded. It now raises a
  `FileNotFoundError` that names `loco-mujoco-myomodel-init`.
- The deprecation warnings of `HumanoidTorque` and `HumanoidMuscle` now name `SkeletonTorque` and
  `SkeletonMuscle` as replacements (they named the deprecated class itself), and the error of
  `LocoEnv.get_default_xml_file_path` names the environment class instead of `type`.
- `Mjx.n_envs` was removed. It returned an attribute that was never set and always raised an `AttributeError`;
  the number of parallel environments is the batch size of the state.
- `RunningStandardization.reset` and `RunningAveragedWindow.reset` only worked for scalar data (the shape was
  passed as the `dtype` argument), and `RunningAveragedWindow.update_state` failed for every shape.
- Setting a path variable (`loco-mujoco-set-*`) failed if the variables file existed but was empty.
- `examples/training_examples/jax_rl_mimic_td3`: `eval.py` and `experiment_resume.py` loaded and trained
  `SACJax` instead of `TD3Jax`, and `conf_resume.yaml` was the SAC configuration.
- `examples/training_examples/jax_distill/conf_traj_swap.yaml` contained absolute paths of a developer machine;
  they are placeholders now.
- Wrong comments in the example configurations: `rnn_type` of S2PG accepts `gru`, `lstm` and `vanilla` (not
  `none`), and the commented domain randomization block of the SuperMimic example is a valid configuration.
- `ImitationFactory.make` without any dataset configuration failed with "Need at least one array to
  concatenate" after building the environment. It now raises a `ValueError` that names the four dataset
  arguments.
- `LAFAN1DatasetConf(dataset_group=...)` could not load any clip, because the names in
  `LAFAN1_LOCOMOTION_DATASETS`, `LAFAN1_DANCE_DATASETS` and `LAFAN1_ALL_DATASETS` carried a `.csv` suffix. The
  suffix was removed from the lists, and `load_lafan1_trajectory` ignores it if it is passed.
- `MetricsHandler` applied the body and site selection to the last axis of the body and site quantities
  (`BodyPosition`, `BodyOrientation`, `BodyVelocity`, `SitePosition`, `SiteOrientation`, `SiteVelocity`,
  `RelSitePosition`, `RelSiteOrientation`, `RelSiteVelocity`), i.e. to the components of a position, rotation
  vector or velocity instead of the bodies or sites. Ids beyond the last component were clamped to it, so with
  the `rel_site_names` of the training examples on the Unitree H1 and G1, `RelSitePosition` and
  `RelSiteOrientation` compared only the third component, counted once per selected site, and `rel_body_names`
  and `rel_site_names` did not restrict `BodyPosition`, `BodyOrientation`, `BodyVelocity`, `SitePosition` and
  `SiteOrientation` to the named elements. The measures are now evaluated on all components of the selected
  bodies and sites. **The values of these nine quantities change**; values logged with earlier versions are not
  comparable with new ones. The joint quantities, the return and the episode length are unchanged. The site
  errors on the benchmark page of the documentation (table of Experiment 1, figure of Experiment 3) were
  computed again with the corrected implementation.
- `validation.rel_joint_names` raised an `IndexError` in `MetricsHandler`, because the joint names were used as
  joint ids. Selecting joints without a free joint, with `rel_joint_names` or by listing the free joint in
  `joints_to_ignore`, failed in `JointPosition` with "cannot reshape array".
- `MetricsHandler` looked up the root body of a site with the site id instead of the id of the body the site
  is attached to (`SiteVelocity`, `RelSiteVelocity`). Both lookups give the same root body for the sites of the
  training examples on the Unitree H1 and G1, so this alone does not change their values.

## [1.2.0] - 2026-11-18

SuperMimic, real-robot deployment through the companion repository
[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real), and trajectories as an explicit
argument of the environment functions.

### Added

- **SuperMimic** (`SuperMimicJax`): whole-body tracking with one differentiable world model. Imagined
  rollouts reuse the environment's own JAX observation, reward and termination functions, so the
  algorithm does not depend on the robot, the observation or the reward. Runs on MJX and MuJoCo Warp.
  Comes with a training example for the Unitree G1 and a per-clip evaluator.
- Goals `TrajFutureFrames`, `RootOriRelTraj` and `TrackingError`, and the reward `SiteTrackingReward`,
  which tracks the mimic sites and needs no robot-specific body list.
- Experimental algorithms in `loco_mujoco.algorithms.experimental`: `S2PGPPOJax`, `BPTTPPOJax`,
  `HistoryPPOJax`, `SACJax`, `TD3Jax` and `VanillaDaggerJax` (DAgger distillation), each with a training
  example.
- Multi-GPU PPO: with `experiment.n_devices` the training function runs under `jax.pmap`.
- `init_agent_state` separates network initialization from training, so that training can be resumed and
  run in chunks.
- `HeightmapTerrain`, a static terrain loaded from a height map, and the `HeightScan` observation.
  All terrains share one height query (`get_height_at`). `recenter_root_xy` controls where episodes start.
- Trajectory handler registry: `RandomStartTrajectoryHandler` (default),
  `RandomTrajFixedStepTrajectoryHandler` and `FixedStartTrajectoryHandler`, selected with
  `th_params=dict(name=...)`.
- Retargeting on first use for robots without hosted data: if a default or LAFAN1 clip is not hosted for
  an environment, it is retargeted from a source environment and cached. The command
  `loco-mujoco-retarget-for-new-env` does this ahead of time.
- Rendering in the browser with [viser](https://viser.studio) (`render_viser`, `mjx_render_viser`,
  `play_trajectory(viser=True)`) and export of animated USD files (`render_usd`, `mjx_render_usd`,
  `play_trajectory(usd=True)`), with Blender scripts in `scripts/blender`.
- `model_contact_conf` changes the contact setup of the robot at construction.
- Under MuJoCo Warp the index of the world is available as `carry.env_id`; `naconmax` is forwarded to the
  simulation data.
- Support for MuJoCo 4.
- `loco-mujoco-download-models` and `loco-mujoco-show-variable` commands.
- Optional dependency groups `viser`, `usd`, `blender` and `test`.
- `GoalTrajMimic` and `MimicReward` accept `sites_for_mimic` and `main_site_name`; robot configurations
  for retargeting can set `joint_limit_solref` and `joint_limit_solimp`.
- A much larger test suite with coverage measurement. Continuous integration runs on pull requests into
  `master` and `dev`, installs the optional extras and reports coverage.
- Tutorials on adding a robot, real-robot deployment, and rendering in the browser and to USD.

### Changed

- **Breaking:** the trajectory is an argument, not environment state. `ImitationFactory.make` returns
  `(env, traj)` and `RLFactory.make` returns `(env, None)`. `reset`, `step`, `mjx_reset` and `mjx_step`
  accept `traj`. Trajectories are padded to fixed shapes, so a compiled training function can be called
  with different clips without recompiling. `mjx_reset(key)` and `mjx_step(state, action)` still work
  without `traj` and then use the trajectory the environment was created with.
- **Breaking:** `loco_mujoco.trajectory` moved to `loco_mujoco.core.trajectory`.
- **Breaking:** `env.load_trajectory(traj)` became `traj = env.process_trajectory(traj)`. The trajectory
  handler holds only the `TrajectoryInfo`; use `TrajectoryHandler.process(traj, model, dt)` to fit a
  trajectory to a model.
- **Breaking:** training functions take the agent state and the trajectory:
  `agent_state = PPOJax.init_agent_state(env, agent_conf, rng)`, then `train_fn(rng, agent_state, traj)`.
  The same holds for GAIL and AMP. `play_policy` takes `traj=traj` for imitation tasks.
- **Breaking:** `create_observation_summary()` returns the path of a local HTML file instead of a URL.
- **Breaking:** the variable `LOCOMUJOCO_CUSTOM_SMPL_CONF_PATH` is now `LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH`.
- Robot models are downloaded on first use (about 200 MB, cached in `~/.loco-mujoco-caches/robot_models`)
  instead of being installed as a git dependency. `LOCOMUJOCO_ROBOT_MODELS_PATH` points to a local copy.
- JAX is restricted to `jax<=0.10.1,!=0.10.0`, because JAX 0.10.0 and 0.10.2 fail to compile the
  SuperMimic training function on the GPU.
- MuJoCo Warp is no longer flagged experimental.
- MuJoCo Warp refuses terrains that write a heightfield per environment (`RoughTerrain`) with an error,
  because it shares heightfield data across worlds.
- `FlatTerrain` is the name of the default flat floor; `terrain_type="StaticTerrain"` keeps working.
- Reward normalization statistics moved from an environment wrapper into the agent state.
- Observation history can be limited to one observation group (`NStepWrapper(group=...)`).
- Environments print their observation summary on creation; `print_obs_summary=False` turns it off.
- Only the `loco_mujoco` package is installed; `docs` and `scripts` are no longer installed as
  top-level packages.
- The `smpl` extra installs `smplx` from PyPI instead of a fork.
- `examples/speed_test.py` is a throughput benchmark across MuJoCo, MJX and MuJoCo Warp.

### Deprecated

- `th_params=dict(random_start=..., fixed_start_conf=...)`. Use
  `th_params=dict(name="FixedStartTrajectoryHandler", start_conf=(0, 0))`.

### Removed

- The wrappers `NormalizeVecReward` and `NormalizeVecRewEnvState`; the algorithms normalize the reward
  themselves.
- `DefaultDatasetConf(dataset_type="pretrained")`; only `"mocap"` exists.
- The console scripts `loco-mujoco-download`, `loco-mujoco-download-real` and
  `loco-mujoco-download-perfect`, which pointed at functions that no longer existed.
- The unused `CollisionExtender`, `optimize_for_collisions` and `MANO_Parser`.

### Fixed

- The reset under MuJoCo Warp restores every per-world field; before, only `qpos` and `qvel` were reset.
- The `Force` observation, `mat2angle_xy` and `angle2mat_xy`, muscle peak-force scaling of the skeleton
  models, `randomize_link_mass=False`, forward kinematics after the initial-state handler, a viewer crash
  when the window is closed, and the `torch_device` argument of the SMPL motion transfer.

### Contributors

Thanks to [mdrolet01](https://github.com/mdrolet01) for the trajectory handler registry, the handler's
backend handling and the quadruped initial-state fix
([#92](https://github.com/robfiras/loco-mujoco/pull/92),
[#95](https://github.com/robfiras/loco-mujoco/pull/95),
[#96](https://github.com/robfiras/loco-mujoco/pull/96)).

## [1.1.0] - 2026-03-10

### Added

- MuJoCo Warp backend (`use_mjwarp=True`) as a second GPU backend next to MJX.
- `save` and `load` functions for PPO and GAIL agents.
- `reset` method in the control function interface.
- Manual trigger for the continuous integration workflow, and continuous integration on the `dev` branch.

### Changed

- MuJoCo, MJX and MuJoCo Warp 3.5 or newer are required.
- Robot models moved to the separate repository
  [loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models) and are no longer part of the
  package.
- Environment construction uses MuJoCo's `MjSpec` interface.
- Stored expected values of the tests were recomputed for the new MuJoCo version.

### Fixed

- Viewer compatibility with newer MuJoCo versions.
- Custom SMPL robot configurations were not found.
- Missing GIFs in the documentation on Read the Docs (Git LFS).

## [1.0.1] - 2025-04-18

A rewrite of the library around JAX. This release is not compatible with the 0.x releases.

### Added

- New simulation core (`loco_mujoco/core`) with a parallel MJX backend next to the serial MuJoCo backend.
  Every environment has a CPU class and an `Mjx` class behind the same interface.
- Modular environments: observations, goals, rewards, control functions, domain randomizers, terrains,
  and initial-state and terminal-state handlers are interchangeable components.
- Task factories `ImitationFactory` and `RLFactory`, and dataset configurations.
- Retargeting pipeline (`loco_mujoco/smpl`): SMPL shape fitting per robot, SMPL-to-robot motion
  retargeting, and robot-to-robot retargeting.
- Datasets: the native LocoMuJoCo dataset and LAFAN1, hosted on the
  [Hugging Face Hub](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets), and AMASS, which is
  retargeted on the user's machine.
- New embodiments: Unitree H1v2, Apptronik Apollo, Booster T1, ToddlerBot, Fourier GR1T2, Unitree Go2,
  Boston Dynamics Spot and ANYmal C. In total 12 humanoid and 4 quadruped environments.
- JAX implementations of PPO, GAIL and AMP, and DeepMimic-style tracking with PPO. Training and
  environment are compiled into one function.
- Trajectory-similarity metrics: Euclidean distance, dynamic time warping and discrete Fréchet distance.
- Domain randomization and terrain randomization.

### Removed

- The interfaces of the 0.x releases: the task strings with fixed datasets per environment, the
  "perfect" datasets with expert actions, and the dependency on MushroomRL for the environments.

## [0.4.1] - 2024-09-13

### Added

- MyoSkeleton environment, based on the MyoSkeleton model by MyoLab.
- Setter for the observation space in the Gymnasium wrapper, and an example that replays actions through
  the Gymnasium interface.
- New datasets.

### Fixed

- Expert dataset generation, a start index of zero being ignored, and further reported issues
  ([#28](https://github.com/robfiras/loco-mujoco/issues/28),
  [#30](https://github.com/robfiras/loco-mujoco/issues/30),
  [#39](https://github.com/robfiras/loco-mujoco/issues/39),
  [#40](https://github.com/robfiras/loco-mujoco/issues/40),
  [#41](https://github.com/robfiras/loco-mujoco/issues/41),
  [#45](https://github.com/robfiras/loco-mujoco/issues/45)).

## [0.3.0] - 2024-05-19

### Added

- Unitree G1 environment with documentation and imitation learning scripts.

### Fixed

- Test datasets updated for the pinned MuJoCo version.

## [0.2.2] - 2024-05-04

### Fixed

- Dataset download paths. This release also contains the changes of 0.2.1, which was published on PyPI
  without a tag.

## [0.2.0] - 2024-04-03

### Added

- Full documentation on Read the Docs, including tutorials for imitation learning.
- Dataset generation pipeline.
- Imitation learning experiment scripts with hyperparameters.
- "Perfect" datasets for the `Atlas.carry` and `HumanoidTorque4Ages` tasks.
- Render modes in the Gymnasium interface
  ([#14](https://github.com/robfiras/loco-mujoco/issues/14)).

### Fixed

- Custom reward interfaces of the robot environments
  ([#15](https://github.com/robfiras/loco-mujoco/issues/15)).
- Loader of the perfect dataset of the Unitree A1
  ([#20](https://github.com/robfiras/loco-mujoco/issues/20)).
- Running datasets.

## [0.1.0] - 2023-12-21

Initial release, described in the
[NeurIPS 2023 Robot Learning Workshop paper](https://arxiv.org/abs/2311.02496).

### Added

- 12 environments and 27 tasks, each with its own datasets; datasets with ground-truth states and
  actions for most tasks.
- Dynamics randomization.
- Gymnasium and MushroomRL interfaces.
- Interface for custom rewards.
- Examples and baselines.

[Unreleased]: https://github.com/robfiras/loco-mujoco/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/robfiras/loco-mujoco/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/robfiras/loco-mujoco/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/robfiras/loco-mujoco/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/robfiras/loco-mujoco/compare/v0.4.1...v1.0.1
[0.4.1]: https://github.com/robfiras/loco-mujoco/compare/v0.3.0...v0.4.1
[0.3.0]: https://github.com/robfiras/loco-mujoco/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/robfiras/loco-mujoco/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/robfiras/loco-mujoco/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robfiras/loco-mujoco/releases/tag/v0.1.0
