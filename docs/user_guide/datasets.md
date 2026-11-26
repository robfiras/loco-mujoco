# Datasets

LocoMuJoCo provides motion capture data for all twelve humanoid embodiments from three sources. The motions
are retargeted to each embodiment, so the same clip name gives the same movement on every robot.

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf

env, traj = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=DefaultDatasetConf(["squat"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]))

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

The first call downloads the three clips. `traj` is a single
[trajectory](trajectories.md) that contains all of them.

## The three sources

| | Default dataset | LAFAN1 | AMASS |
|---|---|---|---|
| Content | 19 locomotion and balance motions provided by the LocoMuJoCo authors | 40 clips of walking, running, dancing, fighting, jumping and getting up | A collection of motion capture databases with thousands of clips |
| Selected with | {class}`~loco_mujoco.task_factories.dataset_confs.DefaultDatasetConf` | {class}`~loco_mujoco.task_factories.dataset_confs.LAFAN1DatasetConf` | {class}`~loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf` |
| Obtained from | Downloaded from Hugging Face on first use | Downloaded from Hugging Face on first use | Downloaded by you from the AMASS website |
| Retargeting | Done in advance for the twelve humanoids | Done in advance for the twelve humanoids | Done on your machine on first use |
| Setup | None | None | [AMASS and SMPL setup](../getting_started/amass.md) |

The pre-retargeted data is hosted in the
[LocoMuJoCo dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) on Hugging Face.
Check the dataset card there and the AMASS and LAFAN1 websites for the license terms that apply to each
source.

### Embodiments with data

Motion data is available for the nine humanoid robots `Apollo`, `Atlas`, `BoosterT1`, `FourierGR1T2`,
`Talos`, `ToddlerBot`, `UnitreeG1`, `UnitreeH1` and `UnitreeH1v2`, and for the three human models
`SkeletonTorque`, `SkeletonMuscle` and `MyoSkeleton`. The `Mjx...` variant of an environment uses the same
data. There is no motion data for the quadrupeds yet; create them with the
[`RLFactory`](task_factories.md).

### Default dataset

The default dataset consists of 19 tasks, each stored as one clip:

`balance`, `highjump`, `highjump2`, `highknee`, `jumpturn`, `onehopforward`, `onestepleft`, `onesteplong`,
`onestepright`, `onestepside1`, `onestepside2`, `random_walk`, `run`, `squat`, `stepinplace1`, `stepinplace2`,
`stepinplace3`, `walk`, `walkturn`.

```python
DefaultDatasetConf("walk")                  # one task
DefaultDatasetConf(["walk", "squat"])       # several tasks
dict(task=["walk", "squat"])                # the same as a dictionary, for configuration files
```

### LAFAN1

The LAFAN1 clips are derived from the
[LAFAN1 retargeting dataset of Unitree](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset),
not from the original LAFAN1 recordings. A clip is named `<motion><take>_subject<id>`:

| Motion | Clips |
|---|---|
| `walk` | `walk1_subject1`, `walk1_subject2`, `walk1_subject5`, `walk2_subject1`, `walk2_subject3`, `walk2_subject4`, `walk3_subject1` to `walk3_subject5`, `walk4_subject1` |
| `run`, `sprint` | `run1_subject2`, `run1_subject5`, `run2_subject1`, `run2_subject4`, `sprint1_subject2`, `sprint1_subject4` |
| `dance` | `dance1_subject1` to `dance1_subject3`, `dance2_subject1` to `dance2_subject5` |
| `jumps` | `jumps1_subject1`, `jumps1_subject2`, `jumps1_subject5` |
| `fight`, `fightAndSports` | `fight1_subject2`, `fight1_subject3`, `fight1_subject5`, `fightAndSports1_subject1`, `fightAndSports1_subject4` |
| `fallAndGetUp` | `fallAndGetUp1_subject1`, `fallAndGetUp1_subject4`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1` |

```python
LAFAN1DatasetConf("walk1_subject1")
LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"])
```

Three lists of clip names are defined in {mod}`loco_mujoco.datasets.humanoids.LAFAN1`:
`LAFAN1_LOCOMOTION_DATASETS` (16 walking and running clips), `LAFAN1_DANCE_DATASETS` (8 clips) and
`LAFAN1_ALL_DATASETS` (all 40). Select one of them by name with the field `dataset_group`, or pass a list
directly:

```python
from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_LOCOMOTION_DATASETS

conf = LAFAN1DatasetConf(dataset_group="LAFAN1_LOCOMOTION_DATASETS")
conf = LAFAN1DatasetConf(LAFAN1_LOCOMOTION_DATASETS[:4])       # the first four clips of the list
```

A group holds many clips; see [Combining sources](#combining-sources) for the limits on the number of samples and clips.

### AMASS

AMASS motions are addressed by their path below the AMASS folder, without the file extension. The first use of
a motion on an embodiment retargets it, which requires the [AMASS and SMPL setup](../getting_started/amass.md).

```python
from loco_mujoco.task_factories import AMASSDatasetConf

AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
                  "KIT/12/WalkInClockwiseCircle11_poses"])
```

`AMASSDatasetConf(dataset_group="AMASS_LOCOMOTION_DATASETS")` selects a predefined list of 241 walking and
running clips, most of them from the KIT database and the rest from ACCAD, BMLrub, EyesJapanDataset and
TotalCapture. These databases have to be present in your AMASS folder. The list is defined in
`loco_mujoco.smpl.const`.

## Finding the available names

The clip names are the file names in the dataset repository, which can be
[browsed online](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main). The files are laid
out as `DefaultDatasets/mocap/<Environment>/<task>.npz` and `Lafan1/mocap/<Environment>/<clip>.npz`. To list
them from Python:

```python
from huggingface_hub import list_repo_files

files = list_repo_files("robfiras/loco-mujoco-datasets", repo_type="dataset")

default_tasks = sorted(f.split("/")[-1].removesuffix(".npz")
                       for f in files if f.startswith("DefaultDatasets/mocap/UnitreeH1/"))
lafan1_clips = sorted(f.split("/")[-1].removesuffix(".npz")
                      for f in files if f.startswith("Lafan1/mocap/UnitreeH1/"))
```

For AMASS, the names are the paths of the `.npz` files in your local AMASS folder.

## Combining sources

All dataset configurations can be given in the same call. The clips are concatenated in the order default,
AMASS, LAFAN1, custom, and the [trajectory handler](trajectories.md) samples episode starts from all of them.

```python
env, traj = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=DefaultDatasetConf(["squat", "stepinplace1"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]),
    amass_dataset_conf=AMASSDatasetConf(["KIT/12/WalkInClockwiseCircle11_poses"]))

print(traj.data.n_trajectories)    # 4
```

```{important}
An environment holds up to 100,000 samples and 100 clips by default. At the control frequency of 100 Hz this is
about 17 minutes of motion; the default `walk` task alone takes most of it. For larger collections, raise the
limits with `th_params=dict(max_n_samples=..., max_n_trajs=...)`, see [trajectories](trajectories.md).
```

## Download, processing and caching

A clip goes through the following steps when it is requested:

1. **Cache lookup.** If a cache folder is set and contains the clip for this embodiment, it is loaded from
   there and the next two steps are skipped.
2. **Download.** Otherwise the file is downloaded from Hugging Face. The `huggingface_hub` library keeps
   downloaded files in its own cache, so this happens once per machine.
3. **Forward kinematics.** The repository stores only joint positions and velocities to keep the files small.
   The body and site positions, orientations and velocities that rewards and observations need are computed
   by replaying the clip in the environment. If a cache folder is set, the completed clip is written to it.
4. **Adaptation to the environment.** The clip is matched to the joints of the environment and resampled to
   its control frequency, as described under [trajectories](trajectories.md). This step runs on every load.

Step 3 takes most of the time. Set the cache folders once to skip it on later runs:

```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

This creates the entries for three folders, `DEFAULT`, `LAFAN1` and `AMASS`, below the given path. Cached
clips are stored per embodiment, for example `DEFAULT/mocap/UnitreeH1/walk.npz` and
`LAFAN1/UnitreeH1/walk1_subject1.npz`. Delete a file to have it rebuilt, for example after changing the model
of an embodiment.

AMASS clips have no download step. They are retargeted from your local AMASS folder on first use and always
written to the AMASS cache folder, together with the body shape fitted to the embodiment. This folder is
therefore required for AMASS. The folder is organized by environment name, so `UnitreeH1` and `MjxUnitreeH1`
are retargeted and stored separately.

### Embodiments without pre-retargeted data

If a clip of the default dataset or of LAFAN1 does not exist in the repository for the requested environment,
which is the case for robots you add yourself, LocoMuJoCo downloads the clip of a source embodiment and
[retargets it](retargeting.md) on the fly. The source is `SkeletonTorque` for the default dataset and
`UnitreeH1v2` for LAFAN1 unless the environment specifies another one. This needs the SMPL part of the
[AMASS and SMPL setup](../getting_started/amass.md). To retarget all clips ahead of time, see
[retargeting](retargeting.md#retargeting-all-datasets-to-an-environment).

### Command-line tools

The paths are stored in the file `LOCOMUJOCO_VARIABLES.yaml` inside the installed package and are managed with
these commands. Each `set` command takes `--path <folder>`.

| Command | Purpose |
|---|---|
| `loco-mujoco-set-all-caches` | Set the cache folders of all three sources below one path |
| `loco-mujoco-set-conv-lafan1-path` | Set only the cache folder for LAFAN1 clips |
| `loco-mujoco-set-conv-amass-path` | Set only the folder for retargeted AMASS clips |
| `loco-mujoco-set-amass-path` | Set the folder that contains the downloaded AMASS data |
| `loco-mujoco-set-smpl-model-path` | Set the folder that contains the SMPL model |
| `loco-mujoco-set-lafan1-path` | Set a folder with raw LAFAN1 data. Not needed for loading datasets, since the LAFAN1 clips are downloaded. |
| `loco-mujoco-show-variable` | Print one variable (`--name NAME`) or all of them (`--all`) |
| `loco-mujoco-add-variable` | Set a variable by name (`--name NAME --value VALUE`), for example `LOCOMUJOCO_CONVERTED_DEFAULT_PATH`, the cache folder of the default dataset |
| `loco-mujoco-remove-variable` | Remove a variable (`--name NAME`) |
| `loco-mujoco-retarget-for-new-env` | Retarget all default and LAFAN1 clips to one environment, see [retargeting](retargeting.md) |
| `loco-mujoco-download-models` | Download the robot models ahead of time |
| `loco-mujoco-myomodel-init`, `loco-mujoco-myomodel-clear` | Download or remove the license-restricted MyoSkeleton model |

## Using your own motion data

Any motion that you can express as joint positions of an embodiment can be used:

- Build a `Trajectory` from your data and pass it with
  {class}`~loco_mujoco.task_factories.dataset_confs.CustomDatasetConf`, as shown in
  [creating custom trajectories](custom_trajectories.md).
- Motion for a different robot can be transferred with
  [robot-to-robot retargeting](retargeting.md).
- [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real#motion-data) contains a worked example for
  a large external dataset: it converts [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed), whose
  motions are already retargeted to the Unitree G1, into trajectory files and trains on them. BONES-SEED is a
  gated dataset with its own license.

## See also

- [Trajectories](trajectories.md) for the data format and how clips are sampled.
- [Retargeting](retargeting.md) for how the motions are mapped to each embodiment.
- [Task factories](task_factories.md) for the dataset configuration classes.
- [Replay a motion](../getting_started/replay_motion.md) for a first look at the data.
- API reference: {mod}`loco_mujoco.task_factories.dataset_confs`,
  {mod}`loco_mujoco.datasets.humanoids.LAFAN1`.
