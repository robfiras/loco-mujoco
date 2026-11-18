# Installation

<ul class="lmj-steps">
  <li class="current">1 · Installation</li>
  <li><a href="first_environment.html">2 · First environment</a></li>
  <li><a href="replay_motion.html">3 · Replay a motion</a></li>
  <li><a href="train_policy.html">4 · Train a policy</a></li>
  <li><a href="next_steps.html">5 · Evaluate and customize</a></li>
</ul>

LocoMuJoCo runs on Linux, macOS and Windows and requires **Python 3.10 or newer**. Training on the GPU
requires Linux with an NVIDIA GPU.

## Install the package

Create a fresh environment, for example with conda:

```bash
conda create -n loco-mujoco python=3.11
conda activate loco-mujoco
```

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
pip install -e .
```

This installs MuJoCo, MJX, MuJoCo Warp and the CPU version of JAX. It is all you need for the next two pages
of this guide.

## Add GPU support

Training runs thousands of environments in parallel on the GPU. Install the CUDA build of JAX on top:

```bash
pip install "jax[cuda12]<=0.10.1,!=0.10.0"
```

```{important}
JAX 0.10.0 and 0.10.2 fail to compile the SuperMimic training function on the GPU, so LocoMuJoCo currently
requires `jax<=0.10.1,!=0.10.0`. Keep the version constraint when you install the CUDA build.
```

Check that JAX sees the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

The output lists a `CudaDevice` if the GPU is used.

## Check the installation

```bash
python -c "import loco_mujoco; print(loco_mujoco.__version__, len(loco_mujoco.get_registered_envs()), 'environments')"
```

This prints the version and the number of registered environments.

## Robot models

The robot models (about 200 MB) are **downloaded automatically** the first time you create an environment and
are cached in `~/.loco-mujoco-caches/robot_models`. On a cluster or in a container you can fetch them ahead of
time:

```bash
loco-mujoco-download-models
```

To use a local copy instead, for example on a machine without network access, set the environment variable
`LOCOMUJOCO_ROBOT_MODELS_PATH` to the `loco_mujoco_models` folder of the
[models repository](https://github.com/robfiras/loco-mujoco-models).

## Optional extras

Install an extra with `pip install -e ".[name]"`. Several can be combined, for example `".[viser,usd]"`.

| Extra | Adds | Needed for |
|---|---|---|
| `viser` | `viser`, `mjviser`, `trimesh` | [Rendering in the browser](../user_guide/rendering.md), for example over SSH |
| `usd` | `usd-core` | [Exporting animations to USD](../user_guide/rendering.md) |
| `smpl` | `torch`, `smplx`, `tqdm`, `joblib` | [AMASS and retargeting](amass.md) |
| `blender` | `usd-core`, `Pillow` | The Blender scripts in `scripts/blender` |
| `test` | `pytest`, `pytest-cov` | [Running the test suite](../development/testing.md) |

The training examples log to [Weights & Biases](https://wandb.ai/site), which is installed separately:

```bash
pip install wandb
```

## Optional models and datasets

The default and LAFAN1 motion capture datasets are downloaded on demand from the
[LocoMuJoCo dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) on Hugging Face.
Nothing needs to be set up for them. Two things are license-restricted and need an extra step:

MyoSkeleton
: The MyoSkeleton model requires accepting its license. Run `loco-mujoco-myomodel-init` once; it shows the
  license and downloads the model.

AMASS
: The AMASS dataset and the SMPL body model must be downloaded from their websites. Follow
  [AMASS and SMPL setup](amass.md).

## Speed up dataset loading

Datasets store only joint positions and velocities. Everything else, such as body positions, is computed with
forward kinematics when a dataset is loaded. Set a cache directory once so that the result is stored and
reused:

```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

## Troubleshooting

No window opens, or rendering fails on a remote machine
: The default viewer needs a display. On a headless machine, pass `headless=True` when creating an environment
  to render off-screen for video recording, or use the [browser viewer](../user_guide/rendering.md).

`jax.devices()` shows only the CPU
: The CUDA build of JAX is not installed, or it does not match the installed NVIDIA driver. Reinstall it with
  the command above and check the [JAX installation notes](https://docs.jax.dev/en/latest/installation.html).

Downloads fail behind a firewall
: Fetch the models on another machine and point `LOCOMUJOCO_ROBOT_MODELS_PATH` to the copy, as described
  under [Robot models](#robot-models).

Next: [create your first environment](first_environment.md).
