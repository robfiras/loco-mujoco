# AMASS and SMPL setup

[AMASS](https://amass.is.tue.mpg.de/) is a large collection of human motion capture recordings in the format of
the [SMPL](https://smpl.is.tue.mpg.de/) body model. LocoMuJoCo retargets AMASS motions to any of its humanoid
embodiments on your machine. Unlike the default and LAFAN1 datasets, neither AMASS nor the SMPL model may be
redistributed, so both have to be downloaded from their websites and registered with LocoMuJoCo once.

You need this setup if you want to

- load AMASS motions with {class}`~loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf`, or
- retarget motions to a robot that has no pre-retargeted data, for example
  [a robot you added yourself](../user_guide/new_robot.md). Robot-to-robot retargeting passes through the SMPL
  model and therefore needs steps 1 to 5 below, but not the AMASS data itself.

The default and LAFAN1 datasets of the built-in humanoids work without any of this.

```{important}
The SMPL steps require an **editable installation** (`pip install -e .`), because the paths are stored in a
file inside the package folder and the conversion script is run from the repository.
```

## What you will set up

| Item | Where it comes from | Registered with |
|---|---|---|
| PyTorch and the `smplx` package | PyPI | `pip install -e '.[smpl]'` |
| SMPL-H body model and MANO hand model | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/download.php) | `loco-mujoco-set-smpl-model-path` |
| AMASS motion data | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/index.html) | `loco-mujoco-set-amass-path` |
| A folder for the retargeted motions | You choose it | `loco-mujoco-set-conv-amass-path` or `loco-mujoco-set-all-caches` |

## Procedure

### 1. Install the additional dependencies

The retargeting pipeline uses the SMPL model through PyTorch. Install the **CPU build** of PyTorch first; the
GPU build competes with JAX for GPU memory and is not needed, since retargeting runs on the CPU by default.

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Then install the SMPL extra. If PyTorch is not installed yet, this command pulls in the GPU build.

```bash
pip install -e '.[smpl]'
```

```{important}
The SMPL software is licensed for non-commercial scientific research purposes only. Read and accept the
[terms and conditions of the SMPL-X repository](https://github.com/vchoutas/smplx) and its
[license](https://github.com/vchoutas/smplx/blob/main/LICENSE) before you run the command above. If you do not
agree with them, do not install or use the model and the software.
```

### 2. Choose the storage folders

Create one folder for the AMASS data and one for the SMPL model, and tell LocoMuJoCo where they are:

```bash
mkdir -p $HOME/amass $HOME/smpl
loco-mujoco-set-amass-path --path $HOME/amass
loco-mujoco-set-smpl-model-path --path $HOME/smpl
```

Retargeted motions and the fitted body shape of each robot are written to a third folder. If you have already
set a common cache folder with `loco-mujoco-set-all-caches`, it is used and you can skip this command.
Otherwise set it explicitly:

```bash
loco-mujoco-set-conv-amass-path --path $HOME/amass_conv
```

Any folder works. The commands store the paths in `loco_mujoco/LOCOMUJOCO_VARIABLES.yaml`; check them at any
time with `loco-mujoco-show-variable --all`.

### 3. Download the AMASS data

1. Register on the [AMASS website](https://amass.is.tue.mpg.de/index.html) and accept its license.
2. On the download page, pick the datasets you want (for example *DanceDB*, *KIT* or *ACCAD*) and download the
   **SMPL-H G** version of each.
3. Extract the archives into the AMASS folder from step 2, one sub-folder per dataset.

A motion is later addressed by its path relative to this folder, without the `.npz` extension. With the layout
below, the motion is called `DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses`.

```text
$HOME/amass/
├── DanceDB/
│   └── DanceDB/
│       └── 20120911_TheodorosSourmelis/
│           └── Capoeira_Theodoros_v2_C3D_poses.npz
└── KIT/
    └── ...
```

### 4. Download the SMPL-H and MANO models

1. Register on the [MANO website](https://mano.is.tue.mpg.de/download.php) and accept its license.
2. Download both of the following:
   - **Extended SMPL+H model**, which contains the body model without hands,
   - **Models & Code**, which contains the hand models.
3. Extract both archives into the SMPL folder from step 2.

The next step expects exactly these three files:

```text
$HOME/smpl/
├── smplh/neutral/model.npz
└── mano_v1_2/models/
    ├── MANO_LEFT.pkl
    └── MANO_RIGHT.pkl
```

### 5. Generate the combined SMPL-H model

LocoMuJoCo loads a single file, `SMPLH_NEUTRAL.pkl`, which combines the body model and the two hand models.
The downloaded files depend on the deprecated `chumpy` package, which only works with old versions of NumPy,
so the conversion runs in a temporary Python 3.10 environment. Use one of the two methods.

`````{tab-set}
````{tab-item} Automatic (conda)
The script `loco_mujoco/smpl/install_smplh.sh` creates a temporary conda environment, runs the conversion and
removes the environment again. Run it from the repository root, with the environment active in which
LocoMuJoCo is installed:

```bash
bash loco_mujoco/smpl/install_smplh.sh
```

It stops with an error if a conda environment named `smplh_model_conversion_env` already exists.
````

````{tab-item} Manual (venv)
Without conda, create the temporary environment yourself. This requires a Python 3.10 interpreter.

```bash
python3.10 -m venv smplh_model_conversion_env
source smplh_model_conversion_env/bin/activate    # Windows: smplh_model_conversion_env\Scripts\activate
pip install 'numpy<1.23.0' chumpy tqdm pyyaml

cd path/to/loco-mujoco
python loco_mujoco/smpl/generate_smplh_model.py --smpl-conf-file loco_mujoco/LOCOMUJOCO_VARIABLES.yaml

deactivate
rm -r smplh_model_conversion_env
```
````
`````

Both methods write `SMPLH_NEUTRAL.pkl` into the SMPL folder.

### 6. Verify the setup

Check that the file was generated and that the three paths are set:

```bash
ls $HOME/smpl/SMPLH_NEUTRAL.pkl
loco-mujoco-show-variable --name LOCOMUJOCO_SMPL_MODEL_PATH
loco-mujoco-show-variable --name LOCOMUJOCO_AMASS_PATH
loco-mujoco-show-variable --name LOCOMUJOCO_CONVERTED_AMASS_PATH
```

Then load and replay an AMASS motion. Replace the path with a motion from a dataset you downloaded:

```python
from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf

env, traj = ImitationFactory.make(
    "UnitreeH1",
    amass_dataset_conf=AMASSDatasetConf(
        ["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"]))

env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)
```

The same example is in `examples/replay_datasets/smpl_example.py`. On the first run for a robot, the log shows
`Robot shape file not found, fitting new one ...`, followed by the retargeting of the motion. This can take a few
minutes. Later runs find the results in the folder for retargeted motions and start immediately.

## Troubleshooting

`Optional smpl depencies not installed`
: The `smpl` extra is missing in the active environment. Repeat step 1.

`KeyError: 'LOCOMUJOCO_SMPL_MODEL_PATH'`, `'LOCOMUJOCO_AMASS_PATH'` or `'LOCOMUJOCO_CONVERTED_AMASS_PATH'`
: The corresponding path is not set. Repeat step 2.

`ValueError: '...' is not in list` while loading a motion
: The motion path does not exist below the AMASS folder. Check the spelling and that the archive was extracted
  into the folder set in step 2; paths are relative to that folder and have no `.npz` extension.

PyTorch runs out of GPU memory
: JAX reserves most of the GPU memory when it starts. Install the CPU build of PyTorch, or set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python.

## See also

- [Datasets](../user_guide/datasets.md) for how AMASS motions are selected, combined with other sources and
  cached.
- [Retargeting](../user_guide/retargeting.md) for what happens to a motion between the AMASS file and the robot.
- [Adding a new robot](../user_guide/new_robot.md) for retargeting to your own embodiment.
