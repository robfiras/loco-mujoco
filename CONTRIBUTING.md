# Contributing to LocoMuJoCo

Contributions are welcome: bug reports, fixes, new robots, new environment components, new algorithms,
and improvements to the documentation. This page describes how the project is developed and what a pull
request should contain.

By taking part in the project you agree to follow the
[code of conduct](https://github.com/robfiras/loco-mujoco/blob/master/CODE_OF_CONDUCT.md).

## Reporting bugs and asking questions

Bugs, feature requests and questions go to the
[GitHub issue tracker](https://github.com/robfiras/loco-mujoco/issues). Search the open and closed issues
first; many installation and dataset questions have been answered there.

A bug report is most useful when it contains:

- the LocoMuJoCo version (`python -c "import loco_mujoco; print(loco_mujoco.__version__)"`) or the commit,
- the versions of Python, `mujoco`, `mujoco-mjx`, `mujoco-warp` and `jax`, and whether JAX runs on CPU or GPU,
- the backend the problem occurs on (MuJoCo on CPU, MJX, or MuJoCo Warp),
- a short script that reproduces the problem, and the full traceback.

Do not report security problems or anything that contains private data in a public issue; contact the
maintainers through GitHub instead.

## Development setup

LocoMuJoCo needs Python 3.10 or newer. Continuous integration uses Python 3.11.

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
pip install -e ".[test]"
```

The `test` extra installs `pytest` and `pytest-cov`. Further extras are optional and only needed when you
work on the corresponding feature:

| Extra | Needed for |
|---|---|
| `smpl` | Retargeting from SMPL and AMASS (`torch`, `smplx`). The SMPL model files have their own license. |
| `viser` | Rendering in the browser. |
| `usd` | Exporting animated USD files. |
| `blender` | The standalone scripts in `scripts/blender/`. |
| `docs` | Building the documentation and checking docstring coverage. |

Continuous integration installs all of them so that no test is skipped:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,usd,viser,smpl,blender]"
```

For GPU training install a CUDA build of JAX, for example `pip install "jax[cuda12]<=0.10.1,!=0.10.0"`.
The tests do not need a GPU.

The robot models are downloaded on first use. To fetch them ahead of time run `loco-mujoco-download-models`.
The [installation guide](https://loco-mujoco.readthedocs.io/en/latest/getting_started/installation.html)
has the details.

## Branches and pull requests

- `master` holds the released code. Each release is a tag on `master`.
- `dev` is the integration branch. **Open pull requests against `dev`.**
- Work on a branch of your fork, named after what it does, for example `fix/init-state-type-key` or
  `feat/trajectory-handler-registry`.

Continuous integration runs the test suite and builds the documentation on every push and pull request to
`master` and `dev`.
A maintainer reviews the pull request and merges it into `dev`; `dev` is merged into `master` when a
release is made (see the
[release process](https://loco-mujoco.readthedocs.io/en/latest/development/releasing.html)).

Keep a pull request to one topic. Recent commit messages follow the
[Conventional Commits](https://www.conventionalcommits.org) pattern (`fix(mjx): ...`, `feat(terrain): ...`,
`docs: ...`); please use it for new commits.

### What a good pull request contains

- **Tests.** New behaviour comes with a test, and a bug fix with a test that fails without the fix.
  See [Testing](https://loco-mujoco.readthedocs.io/en/latest/development/testing.html).
- **Both backends.** Environment components run on NumPy (MuJoCo on CPU) and on JAX (MJX and MuJoCo Warp).
  A new component supports both and is tested on both, unless there is a stated reason why it cannot.
- **Docstrings.** Public classes, methods and functions have a Google-style docstring with types.
- **Documentation.** If the change is visible to users, update the page that describes it, and add new
  public classes to the user guide where they belong.
- **A changelog entry.** Add a line to the "Unreleased" section of
  [CHANGELOG.md](https://github.com/robfiras/loco-mujoco/blob/master/CHANGELOG.md). Mark breaking changes
  as such and say how to migrate.

## Running the tests

```bash
# the whole suite, as far as it runs without network access
pytest --ignore=tests/test_task_factories.py

# one file, one test
pytest tests/test_reward.py
pytest tests/test_reward.py::test_MimicReward
```

`tests/test_task_factories.py` downloads datasets from the Hugging Face Hub and is therefore not part of
continuous integration. Run it locally when you change the task factories or the dataset loaders.

The whole suite in one process needs a lot of memory, because JAX keeps compiled functions alive.
Continuous integration runs one `pytest` process per test file; the
[testing page](https://loco-mujoco.readthedocs.io/en/latest/development/testing.html) shows the loop.

## Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W` turns warnings into errors; continuous integration runs the same command. The conventions for pages and docstrings are described in
[Writing documentation](https://loco-mujoco.readthedocs.io/en/latest/development/documentation.html).

## Adding to the library

Read [Architecture for developers](https://loco-mujoco.readthedocs.io/en/latest/development/architecture.html)
first. It explains the class hierarchy, the order in which components are called during `reset` and `step`,
and the rules that keep the environments compatible with `jax.jit` and `jax.vmap`.

| To add | Read | Where the code goes |
|---|---|---|
| A robot or another embodiment | [Adding a new robot](https://loco-mujoco.readthedocs.io/en/latest/user_guide/new_robot.html) | `loco_mujoco/environments/`, a retargeting configuration in `loco_mujoco/smpl/robot_confs/`, the model files in [loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models) |
| An observation, goal, reward, control function, domain randomizer, terrain, initial-state or terminal-state handler | [Writing custom modules](https://loco-mujoco.readthedocs.io/en/latest/user_guide/custom_modules.html) | the matching sub-package of `loco_mujoco/core/` |
| An algorithm | [Training](https://loco-mujoco.readthedocs.io/en/latest/training/index.html) | `loco_mujoco/algorithms/` (`experimental/` for methods that are not yet benchmarked), with a training example in `examples/training_examples/` |
| A dataset or trajectory source | [Datasets](https://loco-mujoco.readthedocs.io/en/latest/user_guide/datasets.html) and [Custom trajectories](https://loco-mujoco.readthedocs.io/en/latest/user_guide/custom_trajectories.html) | `loco_mujoco/datasets/`, `loco_mujoco/task_factories/` |

A new environment needs a CPU class and an `Mjx` class, both registered in the package `__init__`.
`tests/test_environment_construction.py` iterates over the registry, so it covers a new environment
without changes.
A new algorithm needs a short training run in `tests/test_algorithms.py` and a configuration in
`tests/test_conf/algorithm_confs/`.

## Code conventions

There is no automatic formatter or linter in continuous integration at the moment; only docstring coverage
is checked. Follow the style of the surrounding code:

- **Google-style docstrings** with `Args:`, `Returns:` and, where it applies, `Raises:`. Types are given
  in the signature as type hints and repeated in the docstring.
- **Type hints** on public functions and methods.
- **Lines up to about 120 characters**, four spaces of indentation, imports at the top of the module.
- **Stateless components.** Observations, rewards, terrains and the other components do not store
  per-episode data on `self`. State lives in a `flax.struct.dataclass` that is created in `init_state`,
  returned from every method that changes it, and carried in the environment state.
- **One implementation for both backends.** Component methods receive a `backend` argument, which is either
  `numpy` or `jax.numpy`, and use it for array operations. Code that runs inside `jax.jit` uses
  `jax.lax.cond`, `jax.lax.fori_loop` and `jnp.where` instead of Python control flow on array values.
- **Registration.** Components and environments are selected by class name. Call `YourClass.register()`
  after defining a new one.
- **No new required dependencies** without discussion. Optional features go behind an extra in
  `pyproject.toml` and are imported lazily.

## License

LocoMuJoCo is released under the
[MIT license](https://github.com/robfiras/loco-mujoco/blob/master/LICENSE). By submitting a contribution you
agree that it is licensed under the same terms. Only contribute code, models and data that you have the
right to release under this license; robot models and motion data from third parties keep their own
licenses and are hosted outside this repository.
