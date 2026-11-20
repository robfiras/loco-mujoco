# Release process

This page describes how a LocoMuJoCo release is made and how the repositories that belong to the project
relate to it. Releases are made by the maintainers. The process is short and partly manual; steps that are
a recommendation and not established practice are marked as *recommended*.

## What a release consists of

| Part | Where |
|---|---|
| The version number | `__version__` in `loco_mujoco/__init__.py` |
| A git tag `vX.Y.Z` | on the `master` branch of [robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco) |
| Release notes | a [GitHub release](https://github.com/robfiras/loco-mujoco/releases) for the tag, and a section in the [changelog](changelog.md) |
| The package | [loco-mujoco on PyPI](https://pypi.org/project/loco-mujoco/) |
| The documentation | [loco-mujoco.readthedocs.io](https://loco-mujoco.readthedocs.io) |
| The robot models | a pinned commit of [robfiras/loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models) |
| The datasets | the Hugging Face repository [robfiras/loco-mujoco-datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) |

## Version number

The version is defined once, as `__version__` in `loco_mujoco/__init__.py`. Everything else reads it from
there: `setup.py` imports it for packaging (`pyproject.toml` declares the version as `dynamic`), and
`docs/conf.py` uses it for the documentation.

Versions have the form `MAJOR.MINOR.PATCH`. The history so far:

- `0.x` releases: the CPU benchmark described in the 2023 workshop paper.
- `1.0`: the rewrite around JAX, which was not compatible with `0.x`.
- Minor releases since then (`1.1`, `1.2`) added features. Version 1.2.0 also changed interfaces; those
  changes are listed under "Changed" and marked as breaking in the [changelog](changelog.md).

*Recommended:* treat the major version as the compatibility promise from now on. A change that breaks
user code goes into a minor release only with a migration table in the changelog and, where possible, a
deprecation period in which the old call keeps working with a `DeprecationWarning`, as was done for the
`th_params` keys in 1.2.0.

## Branches

- `dev` is where work is integrated. Pull requests target `dev`
  (see [Contributing](contributing.md)).
- `master` is the released state. `master` is merged into `dev` when needed, and at a release `master`
  is moved to the state of `dev`; the tag `v1.2.0`, for example, is on a commit that is the head of both.

Continuous integration runs on both branches and on pull requests into them.

## Making a release

1. **Check that `dev` is ready.** Continuous integration passes on `dev`. Run
   `pytest tests/test_task_factories.py` locally, because this file needs the hosted datasets and is not
   part of continuous integration (see [Testing](testing.md)). *Recommended:* run one training example
   on a GPU, since continuous integration only runs on the CPU; the upper bound on the JAX version in
   `pyproject.toml` exists because two JAX releases failed to compile a training function on the GPU only.
2. **Check the model pin.** If the robot models changed, `MODELS_REVISION` in
   `loco_mujoco/core/utils/models.py` must point to the commit of the models repository that this release
   was tested with (see below).
3. **Update the changelog.** Rename the "Unreleased" section of `CHANGELOG.md` to the new version with
   the date, add an empty "Unreleased" section above it, and update the comparison links at the bottom of
   the file.
4. **Bump the version** in `loco_mujoco/__init__.py`. *Recommended:* also update `version` and
   `date-released` in `CITATION.cff`.
5. **Merge `dev` into `master`** and push.
6. **Tag** the commit on `master` and push the tag:

   ```bash
   git tag v1.3.0
   git push origin v1.3.0
   ```

   Tags are named `v` followed by the version. The existing tags are lightweight tags.
7. **Publish the GitHub release** for the tag. The release notes are written by hand: a summary of the
   highlights, the changes grouped by area, and a table of breaking changes with the old and the new call.
   The changelog section is the basis.
8. **Build and upload the package** (next section).
9. **Check the documentation** on Read the Docs (section below).

## Packaging

The package is built with setuptools. The relevant files:

| File | Role |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, optional extras, console scripts. Only the `loco_mujoco` package is included. |
| `setup.py` | Supplies the version and the package data (the files of `loco_mujoco/smpl`, such as the retargeting configurations). |
| `MANIFEST.in` | Keeps `tests/`, `examples/`, `.github/` and any local `loco_mujoco/models` out of the source distribution. |
| `Makefile` | Shortcuts for building and uploading. |

The `Makefile` targets:

```bash
make clean      # remove dist/ and build/
make package    # python3 setup.py sdist
make install    # pip install the built source distribution, to test it
make upload     # python3 -m twine upload dist/*
```

`make upload` needs [twine](https://twine.readthedocs.io) and PyPI credentials of a maintainer.

`make package` builds a source distribution only. The 1.2.0 upload on PyPI contains a wheel as well.
*Recommended:* build both with [build](https://build.pypa.io) and check them before uploading:

```bash
python -m pip install build twine
python -m build              # writes dist/*.tar.gz and dist/*.whl
python -m twine check dist/*
python -m twine upload dist/*
```

Then test the upload in a fresh environment: `pip install loco-mujoco`, create an environment with
`RLFactory.make`, and confirm that the robot models are downloaded.

Not every tag is on PyPI and not every PyPI version is a tag: 1.0.1 and 1.1.0 were released on GitHub only,
while 0.2.1 and 1.0.0 exist on PyPI without a tag. *Recommended:* keep the two in step from now on, one
upload per tag.

The examples, tutorials and tests are not part of the PyPI package. Users who want them clone the
repository, which is why the installation guide offers an editable install next to `pip install loco-mujoco`.

## Documentation

[Read the Docs](https://loco-mujoco.readthedocs.io) builds the documentation from `docs/` according to
`.readthedocs.yaml`: Python 3.11, the package itself and `docs/requirements.txt`. The build fetches the
Git LFS files of the repository, because some images are stored with LFS.

The links in the README, in `CONTRIBUTING.md` and in the release notes point to the `latest` version.
Which branches and tags are built is configured in the Read the Docs project settings, not in the
repository. *Recommended:* activate a version for each release tag and keep `stable` pointing to the
newest tag, so that users of a released version can read the documentation that matches it.

When pages are moved or renamed, add an entry to `redirects` in `docs/conf.py`, so that links in old
release notes and in papers keep working.

## Robot models

The model files (XML, meshes, textures; about 200 MB) are in the separate repository
[loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models), so that the Python package stays
small. Since 1.2.0 they are not a dependency but are downloaded on first use.

A release is tied to the models through `MODELS_REVISION` in `loco_mujoco/core/utils/models.py`, the full
hash of one commit of the models repository. The library downloads exactly that commit, and the cache
directory is named after it, so different LocoMuJoCo versions on one machine do not interfere. The
consequences for development:

- A change to a model is made in the models repository first. It reaches users only when
  `MODELS_REVISION` is updated in LocoMuJoCo, in the same pull request as any code or test change that
  depends on it.
- Changing a model can change stored expected values of the tests; regenerate them as described in
  [Testing](testing.md).
- The cache of the models in continuous integration is keyed on the hash of `models.py`, so a new pin
  triggers a fresh download there.
- Commits of the models repository that are pinned by a release must stay reachable. Do not rewrite the
  history of that repository.

The MyoSkeleton model is not in the models repository. `loco_mujoco/utils/myomodel_init.py` pins a commit
of [myolab/myo_model](https://github.com/myolab/myo_model) in the same way.

## Datasets

The retargeted default dataset and LAFAN1 are hosted in the Hugging Face dataset repository
[robfiras/loco-mujoco-datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets), in the
folders `DefaultDatasets/mocap/<Environment>/` and `Lafan1/mocap/<Environment>/`. The library downloads
single files from it on demand. AMASS is not hosted; it is retargeted on the user's machine.

Unlike the models, the dataset repository is **not pinned** to a revision: every LocoMuJoCo version
downloads the current files. Changes to the dataset repository therefore take effect for all versions at
once. In practice:

- Add files; do not rename or remove files that a released version loads.
- A file that is replaced changes the data for existing users. Users who cached the clip (in the Hugging
  Face cache or in `LOCOMUJOCO_CONVERTED_DEFAULT_PATH` / `LOCOMUJOCO_CONVERTED_LAFAN1_PATH`) keep the old
  one. Mention such a change in the changelog.
- For a robot without hosted data the library retargets the clips locally on first use, so a new
  environment can be released before its data is uploaded.

*Recommended:* tag the dataset repository at each release, so that the data a release was evaluated with
can be recovered.

## Companion repository

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) contains the sim-to-real tooling and
depends on LocoMuJoCo. It is versioned separately. After a release with interface changes, check that
its training and deployment scripts still run against the new version.
