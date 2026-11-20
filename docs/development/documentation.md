# Writing documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org), the
[PyData theme](https://pydata-sphinx-theme.readthedocs.io) and [MyST Markdown](https://myst-parser.readthedocs.io).
The sources are in `docs/`, and [Read the Docs](https://loco-mujoco.readthedocs.io) builds them on every push.

## Building locally

```bash
pip install -e .
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The first build takes a minute or two because the API
reference imports the whole package; later builds are incremental.

Continuous integration builds the documentation with warnings treated as errors (`-W`), so a broken
cross-reference or a malformed docstring fails the build. Run the same command before opening a pull request:

```bash
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Layout

| Folder | Contents |
|---|---|
| `getting_started/` | The linear onboarding track. Short pages, one outcome each. |
| `user_guide/` | Concepts, one page per pluggable component, and guides for extending the library. |
| `training/` | One page per algorithm, plus configuration and evaluation. |
| `environments/` | The environment gallery. Generated, see below. |
| `sim2real/` | Real-robot deployment. |
| `api/` | The API reference. Generated from docstrings by `autosummary`; only `index.md` is written by hand. |
| `development/` | Contributor documentation and project pages. |

## Conventions

**Write for a reader who has not seen the code.** State what something is for before how it works, and show
a runnable snippet early.

**Pull code from the repository instead of pasting it.** Tutorial code lives in `examples/` and is included with
`literalinclude`, so the documentation cannot drift from code that is tested:

````markdown
```{literalinclude} ../../examples/tutorials/01_creating_mujoco_env.py
:language: python
```
````

Short inline snippets are fine for illustrating a single call. They must be correct for the current API.

**Cross-reference the API** instead of naming a class in plain code font. The tilde shortens the displayed name
to its last component:

```markdown
{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`
{meth}`~loco_mujoco.environments.base.LocoEnv.play_trajectory`
{mod}`loco_mujoco.core.reward`
```

**Link to other pages with relative Markdown links**, for example `[datasets](../user_guide/datasets.md)`.

**Use admonitions sparingly**, for information the reader would otherwise miss:

````markdown
```{note}
The MyoSkeleton model requires accepting its license first.
```
````

Available kinds are `note`, `tip`, `important`, `warning` and `seealso`.

**Use tabs for alternatives** such as CPU and GPU, or MuJoCo and MJX:

`````markdown
````{tab-set}
```{tab-item} MuJoCo (CPU)
...
```
```{tab-item} MJX (GPU)
...
```
````
`````

**Images** that are shared with the README live in `imgs/` at the repository root and are referenced with a
relative path, for example `../../imgs/terrains.gif`.

**Headings** use sentence case. Each page has exactly one top-level heading.

## The environment gallery

The pages under `docs/environments/` are generated from the environment registry by
`docs/_scripts/generate_environments.py`. It instantiates every environment and writes one page per embodiment
with its observation and action space. Rerun it after adding an environment or changing a default observation
or action space, and commit the result:

```bash
python docs/_scripts/generate_environments.py
```

## Docstrings

The API reference is only as good as the docstrings. Public classes, methods and functions have a
Google-style docstring:

```python
def interpolate(self, new_frequency: float) -> "TrajectoryData":
    """
    Resample the trajectory to a new frequency.

    Args:
        new_frequency (float): Target frequency in Hz.

    Returns:
        TrajectoryData: The resampled trajectory.

    """
```

Docstring coverage is checked in continuous integration with
[interrogate](https://interrogate.readthedocs.io); the threshold is set in `pyproject.toml`.
