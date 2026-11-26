# Testing

LocoMuJoCo has a [pytest](https://docs.pytest.org) suite in `tests/`. Apart from one file it runs without
network access, without a GPU and without a display. Continuous integration runs it on every push and
pull request to `master` and `dev`.

## Running the tests

Install the package with the test dependencies:

```bash
pip install -e ".[test]"
```

Then run the suite from the repository root:

```bash
pytest --ignore=tests/test_task_factories.py
```

A single file or a single test:

```bash
pytest tests/test_reward.py
pytest tests/test_reward.py::test_MimicReward
pytest tests/test_observation.py -k "BodyPos"
```

Three things are worth knowing before the first run.

**The robot models are needed.** They are downloaded on first use; run `loco-mujoco-download-models` once
to fetch them ahead of time. Tests that only use the small test model in `tests/test_conf/` do not need them.

**`tests/test_task_factories.py` needs the network.** It creates every environment with the hosted default
and LAFAN1 datasets, and already lists the files of the Hugging Face dataset repository when it is
imported. It is therefore excluded from continuous integration. Run it locally after changing the task
factories or the dataset loaders:

```bash
pytest tests/test_task_factories.py
```

**Tests for optional features are skipped if the extra is missing.** The SMPL, viser, USD and Blender
tests call `pytest.importorskip`. To run everything, install the extras as continuous integration does:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,usd,viser,smpl,blender]"
```

A few tests remain skipped where the software is missing: the MuJoCo Warp terrain test needs a GPU, and
the Blender integration test needs a Blender executable.

### Memory, and why CI runs one process per file

JAX keeps every compiled function alive in process-wide caches. The suite compiles many environments and
training functions, so the memory of a single `pytest` process grows with the number of tests. Two
measures keep this in check:

- `tests/conftest.py` has an `autouse` fixture that calls `jax.clear_caches()` and `gc.collect()` after
  every test.
- Continuous integration starts one `pytest` process per test file and appends the coverage data, because
  the whole suite in one process peaks close to the 16 GB of the runner.

To reproduce the CI run locally, including the coverage report:

```bash
status=0
for f in tests/test_*.py; do
  [ "$f" = tests/test_task_factories.py ] && continue
  pytest "$f" --cov=loco_mujoco --cov-append --cov-report= || status=1
done
python -m coverage report --skip-covered
python -m coverage html        # writes htmlcov/index.html
```

Delete `.coverage` before the loop if an older run left one behind, because `--cov-append` adds to it.

Most test modules set `jax.config.update("jax_platform_name", "cpu")`, so the suite runs on the CPU even
on a machine with a GPU.

## What the suite covers

Without the task-factory file the suite has more than 500 tests (545 collected at version 1.2.0, counting
each parametrization). Many tests are parametrized over `backend in ["numpy", "jax"]` and check the same
behaviour on both backends.

| Area | Test files | What is checked |
|---|---|---|
| Simulation core | `test_mjx.py`, `test_environment_construction.py`, `test_model_files.py`, `test_observation_summary.py`, `test_test_conf.py` | MJX and MuJoCo Warp reset and step, including the reset inside a step; construction, reset and step of every registered environment on the CPU and construction of every `Mjx` environment; locating and downloading the robot models; the observation summary. |
| Observations and goals | `test_observation.py`, `test_observation_extra.py`, `test_goals.py`, `test_goals_visualization.py` | Every observation type against values read from MuJoCo, on both backends; observation groups and indices; goals and their visualization. |
| Rewards | `test_reward.py`, `test_reward_branches.py` | Every reward against stored expected values on both backends; the individual penalty terms of `LocomotionReward`. |
| Control, randomization, terrain | `test_control_functions.py`, `test_domain_randomizer.py`, `test_terrain.py` | Default and PD control; each switch of the default randomizer; the terrain interface, height queries, `HeightmapTerrain` and `RoughTerrain`. |
| Initial and terminal states | `test_initial_state_handler.py`, `test_quadruped_init_state.py`, `test_terminal_state_handler.py` | Initial states against stored expected values; the standing pose of the quadrupeds; each terminal-state handler. |
| Trajectories and datasets | `test_trajectory.py`, `test_trajectory_handler.py`, `test_default_traj_offline.py`, `test_imitation_factory_offline.py`, `test_lafan1_load_offline.py`, `test_play_trajectory_offline.py`, `test_dataset_variables.py`, `test_data_generation_utils.py` | The trajectory dataclasses (saving, loading, concatenation, interpolation, conversion between NumPy and JAX); the handler registry and the start each handler picks; the cache paths of the dataset loaders; trajectory replay; the path variables. |
| Retargeting | `test_smpl_parser.py`, `test_smpl_retargeting.py`, `test_skeleton_scaling.py` | The SMPL parser and the retargeting functions with synthetic SMPL-H models; body scaling of the skeleton models. |
| Algorithms | `test_algorithms.py`, `test_supermimic.py`, `test_offpolicy_helpers.py`, `test_replay_buffer.py` | For every algorithm: that the training function compiles, saving and loading an agent, and evaluation; algorithm options; short SuperMimic training runs and its building blocks; the replay buffer; the trajectory metrics. |
| Wrappers | `test_gymnasium_wrapper.py`, `test_wrappers.py` | The Gymnasium interface; the observation-history wrapper. |
| Rendering and export | `test_video_recorder.py`, `test_viewer_camera.py`, `test_viser_viewer.py`, `test_usd_exporter.py`, `test_blender_scripts.py`, `test_blender_integration.py` | Headless video recording, the viewer's camera helper, the viser viewer, USD export and the Blender scripts. |
| Utilities | `test_math.py`, `test_core_utils_helpers.py` | Rotation and quaternion helpers and other small functions. |
| Task factories (not in CI) | `test_task_factories.py` | Every environment with the hosted default and LAFAN1 datasets and a custom trajectory. Needs network access. |

## Test infrastructure

### `tests/conftest.py`

Two fixtures apply to every test automatically:

- `_release_jax_caches` clears JAX's compilation caches after each test (see above).
- `offline_default_traj` replaces `ImitationFactory.get_default_traj` with a function that builds a
  trajectory from the environment's own MuJoCo model: one forward-kinematics frame repeated over time.
  The joints, bodies and sites match the environment exactly, so the trajectory passes through
  `process_trajectory`, the mimic goals and the rewards like a downloaded one. This is why tests of the
  imitation algorithms need no network. The synthetic clip is a standing pose and is not meant to be
  realistic; it exercises code paths, not learning performance.

### `tests/test_conf/`

This package is imported by most test modules with `from test_conf import *`.

| File | Contents |
|---|---|
| `humanoid_test.xml` | A small humanoid model that ships with the tests. |
| `dummy_humanoid_env.py` | `DummyHumamoidEnv`, a {class}`~loco_mujoco.environments.base.LocoEnv` on that model. The argument `enable_mjx` selects the backend. It can roll out a nominal trajectory and return the transitions, which is how rewards are evaluated on fixed states. |
| `fixtures.py` | Fixtures for trajectory data on both backends, `standing_trajectory` and `falling_trajectory` (recorded from the dummy environment), `mock_random`, and one fixture per algorithm configuration. |
| `trajectory_generator.py` | `generate_test_trajectories(expert_traj, nominal_traj, backend, ...)`, which creates the dummy environment with the given components, loads the expert trajectory and replays the nominal one. |
| `algorithm_confs/` | One small YAML configuration per algorithm, loaded by the fixtures. |
| `utils.py` | Lists of environment names per backend, and of environments that have hosted datasets. |

The `mock_random` fixture makes tests of stochastic components deterministic. It replaces
`jax.random.normal`, `uniform` and `randint` and their `numpy.random` counterparts with functions that
return fixed values (for example 0.5 for a normal sample, and 30 % of the range for a uniform sample).
With it, a reward with a random goal or a randomized initial state gives the same number on every machine
and on both backends.

### Stored expected values

Several tests compare against numbers stored in the test file. `test_reward.py`, for example, asserts the
sum of a reward over a rollout and its value at step 42, once for NumPy and once for JAX:

```python
assert isclose(reward_sum, 18.869667053222656, abs_tol=_ATOL)
assert isclose(reward_42, 0.1381128579378128, abs_tol=_ATOL)
```

These numbers depend on the physics engine. They change when MuJoCo changes its numerics, when the test
model changes, or when a reward is changed on purpose. To regenerate them, run from the repository root:

```bash
python tests/compute_expected_values.py
```

The script applies the same random mocks as `mock_random`, rebuilds the standing and falling trajectories,
and prints for both backends:

- the reward sum and the reward at step 42 for `TargetXVelocityReward`, `TargetVelocityGoalReward`,
  `LocomotionReward`, `TargetVelocityTrajReward` and `MimicReward`,
- the initial observations produced by `DefaultInitialStateHandler` and `TrajInitialStateHandler`.

Copy the printed values into `tests/test_reward.py` and `tests/test_initial_state_handler.py`. Before
doing so, make sure the change is intended: a reward value that changed without a change to the reward or
to MuJoCo is a regression, not a reason to update the number. State in the pull request why the values
were updated.

The tolerance is a constant `_ATOL` at the top of each of the two files (`1e-5` for the rewards, to absorb
floating-point differences between machines, and `1e-7` for the initial states).

Other test files with stored values, such as `test_observation.py` and `test_domain_randomizer.py`, are not
covered by the script; their values are updated by hand from the failing assertion.

## Coverage

Coverage is measured with [coverage.py](https://coverage.readthedocs.io) through `pytest-cov`. The
configuration is in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`; it measures
line coverage of the `loco_mujoco` package. At version 1.2.0 the suite reached 93 % line coverage.

Some code cannot run in an automated, offline setting and is excluded from the measurement. Each exclusion
is listed in `pyproject.toml` with its reason:

| Excluded | Reason |
|---|---|
| `core/visuals/viewer.py` | The interactive OpenGL viewer opens a window and needs a display. |
| All `__init__.py` files | Imports and registration calls without logic of their own. |
| `utils/retarget_for_env.py` | Command line tool that downloads clips from the Hugging Face Hub and needs the SMPL model. |
| `smpl/generate_smplh_model.py` | Reads the license-restricted SMPL and MANO model files. |
| `utils/myomodel_init.py`, `environments/humanoids/myoskeleton.py`, `myoskeleton_mjx.py` | Need the MyoSkeleton model, which is behind a license prompt. |
| Functions named `play_policy` | They render every step and need an OpenGL context. |
| `load_retargeted_amass_trajectory`, `retarget_traj_from_robot_to_robot` | Entry points that need the AMASS dataset and the SMPL model. The functions they delegate to are tested with synthetic SMPL-H models. |
| `__repr__`, `if TYPE_CHECKING:`, `raise NotImplementedError`, abstract methods, `if __name__ == "__main__":`, `# pragma: no cover` | Standard exclusions for lines that are not meant to be executed by tests. |

Do not add an exclusion to make a number go up. An exclusion is justified when the code needs a display, a
licensed asset or a large download; otherwise write a test, if necessary with a synthetic input as the SMPL
tests do.

## Continuous integration

The workflow is `.github/workflows/continuous_integration.yml`. It runs on pushes and pull requests to
`master` and `dev`, and can be started by hand from the Actions tab. It has three jobs on `ubuntu-latest`:
the test job and the documentation job with Python 3.11, and the installation job with every supported
Python version.

The test job

1. installs a CPU build of PyTorch and the package with the extras `test`, `usd`, `viser`, `smpl` and
   `blender`, so that the tests of the optional features run instead of being skipped,
2. restores the robot models from a cache keyed on `loco_mujoco/core/utils/models.py` (the file that pins
   the model revision) and runs `loco-mujoco-download-models`,
3. runs one `pytest` process per test file with coverage, skipping `tests/test_task_factories.py`,
4. writes the coverage table to the job summary and uploads `coverage.xml` and the HTML report as the
   artifact `coverage-report`.

The job fails if any test file fails. There is no coverage threshold that fails the build, and no lint
step at present.

The documentation job installs the `docs` extra, checks docstring coverage with
[interrogate](https://interrogate.readthedocs.io), and builds the documentation with warnings treated as
errors. See [Writing documentation](documentation.md).

The installation job checks the package as users get it from PyPI, on Python 3.10, 3.11, 3.12 and 3.13. For
each version it

1. builds the wheel and the source distribution with `python -m build` and checks them with
   `twine check --strict`,
2. installs the wheel, not an editable checkout, into a clean virtual environment, so that pip resolves the
   dependency versions a new user would get that day,
3. runs `.github/scripts/install_smoke_test.py` from outside the repository on the CPU: a MuJoCo
   environment, an MJX environment and an imitation environment with a clip of the default dataset.

A missing data file in the wheel, a Python version that the dependencies no longer support, or a new
dependency release that breaks the package all fail this job. It also runs every Monday on its own, without
the other two jobs, so that such a release is noticed between pull requests. To run the smoke test locally,
install the package into a fresh environment and call the script from a directory outside the repository:

```bash
python -m build
python -m venv /tmp/lmj-venv && /tmp/lmj-venv/bin/pip install dist/*.whl
cd /tmp && JAX_PLATFORMS=cpu /tmp/lmj-venv/bin/python /path/to/loco-mujoco/.github/scripts/install_smoke_test.py
```

## Adding tests

Put new tests in `tests/test_<area>.py`; pytest collects every function whose name starts with `test_`.

**For a new component** (observation, reward, terrain, ...), use the dummy environment. It is small, needs
no download and runs on both backends:

```python
import numpy as np
import jax
import pytest

from test_conf import DummyHumamoidEnv

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_my_reward_is_finite(backend):
    env = DummyHumamoidEnv(enable_mjx=backend == "jax", reward_type="MyReward",
                           reward_params=dict(my_param=1.0), **DEFAULTS)
    key = jax.random.PRNGKey(0)
    action = np.zeros(env.info.action_space.shape)

    if backend == "numpy":
        env.reset(key)
        _, reward, _, _, _ = env.step(action)
    else:
        state = env.mjx_reset(key)
        reward = env.mjx_step(state, action).reward

    assert np.isfinite(reward)
```

Good component tests check, in this order of preference: a value that can be derived independently (for
example from `MjData` directly, as the observation tests do); agreement between the NumPy and the JAX
result; and, where neither is possible, a stored expected value. If the component is random, request the
`mock_random` fixture. If it needs a trajectory, use `standing_trajectory` and `falling_trajectory` with
`generate_test_trajectories`. For a reward or an initial-state handler with stored values, add it to
`tests/compute_expected_values.py` so that the values can be regenerated.

**For a new environment**, no new test is needed for the basics: `test_environment_construction.py`
iterates over `LocoEnv.registered_envs`, so a registered environment is constructed, reset and stepped on
the CPU, and its `Mjx` class is constructed, automatically. Add tests for anything specific to the robot,
as `test_quadruped_init_state.py` and `test_skeleton_scaling.py` do. If the environment needs an asset that
cannot be downloaded in continuous integration, add it to the `_SKIP` set in that file and say why.

**For a new algorithm**, add a small configuration to `tests/test_conf/algorithm_confs/`, a fixture that
loads it to `tests/test_conf/fixtures.py`, and tests in `tests/test_algorithms.py` that build and trace the
training function and save and load an agent. Keep the number of environments and steps small; the
compilation dominates the run time.

**For code that needs the network or a licensed asset**, test the logic around it with a synthetic input or
with `monkeypatch`, as `test_model_files.py` (model download), `test_lafan1_load_offline.py` (dataset cache)
and `test_smpl_parser.py` (synthetic SMPL model) do.

Before opening a pull request, run at least the test files of the area you changed, and
`tests/test_task_factories.py` if you touched the factories or dataset loaders.
