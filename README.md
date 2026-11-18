<p align="center">
  <img width="70%" src="https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a">
</p>

![continuous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![Documentation Status](https://app.readthedocs.org/projects/loco-mujoco/badge/?version=latest)](https://loco-mujoco.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![PyPI]&#40;https://img.shields.io/pypi/v/loco-mujoco&#41;]&#40;https://pypi.org/project/loco-mujoco/&#41;)

> 🚀 **Latest News:**
> A **new release (v1.2)** just dropped! 🎉  
> **New:** LocoMuJoCo now includes **SuperMimic** and supports **real-robot deployment** on the
> **AgiBot X2 Ultra** and the **Unitree G1** through
> [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real).


**LocoMuJoCo** is an **imitation learning benchmark** specifically designed for **whole-body control**.  
It features a diverse set of environments, including **quadrupeds**, **humanoids**, and **(musculo-)skeletal human models**,
each provided with comprehensive datasets.

Although primarily focused on imitation learning, LocoMuJoCo also supports custom reward function classes,  
making it suitable for pure reinforcement learning as well.

<div align="center">
  <img src="imgs/main_lmj.gif"/>
</div>

### Key Advantages 
✅ Supports **MuJoCo** (single environment) and **MJX/MJWarp** (parallel environments) \
✅ Includes **12 humanoid and 4 quadruped environments**, featuring 4 **biomechanical human models** \
✅ Clean single-file JAX algorithms for quick benchmarking (**SuperMimic**, **PPO**, **GAIL**, **AMP**, **DeepMimic**)\
✅ Combined training and environment into one JIT‑compiled function for lightning‑fast training 🚀 \
✅ **Motion capture datasets** (AMASS, LAFAN1, native LocoMuJoCo) retargeted for each humanoid \
✅ **Robot-to-robot retargeting** allows to retarget any existing dataset from one robot to another \
✅ Powerful **trajectory comparison metrics** including dynamic time warping and discrete Fréchet distance, all in JAX \
✅ Built-in **domain and terrain randomization** \
✅ **Real-robot deployment** with [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real): deployable training setups for PPO and SuperMimic, and the tools to take a policy to the robot \
✅ Modular design: define, swap, and reuse components like observation types, reward functions, terminal state handlers, and domain randomization \
✅ [Documentation](https://loco-mujoco.readthedocs.io/)

---

## Installation

[//]: # (You have the choice to install the latest release via PyPI by running )

[//]: # ()
[//]: # ()
[//]: # (```bash)

[//]: # ()
[//]: # (pip install loco-mujoco )

[//]: # ()
[//]: # (```)

LocoMuJoCo requires **Python 3.10 or newer**. We recommend a fresh environment, for example with conda:

```bash
conda create -n loco-mujoco python=3.11
conda activate loco-mujoco
```

Clone this repo and do an editable installation:

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
pip install -e . 
```

By default, this installs the CPU version of JAX. If you want to use JAX on the GPU, you need to install the following:

```bash
pip install "jax[cuda12]<=0.10.1,!=0.10.0"
```

JAX 0.10.0 and 0.10.2 fail to compile SuperMimic's training function on the GPU, so LocoMuJoCo
currently requires `jax<=0.10.1,!=0.10.0`.

The robot models (about 200 MB) are downloaded automatically the first time you create an environment
and cached in `~/.loco-mujoco-caches/robot_models`. On a cluster or in a container you can fetch them ahead of time:

```bash
loco-mujoco-download-models
```

To use a local copy instead, for example on a machine without network access, point `LOCOMUJOCO_ROBOT_MODELS_PATH`
to the `loco_mujoco_models` folder of the [models repository](https://github.com/robfiras/loco-mujoco-models).

> [!NOTE]
> If you want to run the **MyoSkeleton** environment, you need to additionally run
> `loco-mujoco-myomodel-init` to accept the license and download the model.


### Datasets

LocoMuJoCo provides three sources of motion capture (mocap) data for humanoid environments: default (provided by us), LAFAN1, and AMASS. The first two datasets
are available on the [LocoMuJoCo HuggingFace dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets)
and will be downloaded and cached automatically for you. AMASS needs to be downloaded and installed separately due to
their licensing. See [here](loco_mujoco/smpl) for more information about the installation.

Other datasets can be used as well. [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real#motion-data) shows
how to load your own trajectories and how to convert and train on [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed).

This is how you can visualize the datasets:

```python
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


# # example --> you can add as many datasets as you want in the lists!
env, traj = ImitationFactory.make("UnitreeH1",
                            default_dataset_conf=DefaultDatasetConf(["squat"]),
                            lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            #amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"])
                            )

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

#### Speeding up Dataset Loading
LocoMuJoCo only stores datasets with joint positions and velocities to save memory. All other attributes are calculated 
using forward kinematics upon loading. If you want to speed up the dataset loading, you can define caches for the datasets. This will
store the forward kinematics results in a cache file, which will be loaded on the next run: 

```bash
loco-mujoco-set-all-caches --path <path to cache>
```

For instance, you could run:
```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

---

## Using MJX with the MjWarp backend

LocoMuJoCo’s MJX environments can run either on the standard MJX backend or on the **MjWarp** backend.
Switching to MjWarp can further accelerate parallel simulation on compatible GPUs.

- **From the low-level MJX API**, pass `use_mjwarp=True` (and optionally `nconmax` / `njmax`) when creating an `Mjx` env:

```python
from loco_mujoco.core import Mjx

mjx_env = Mjx(
    spec=spec,
    actuation_spec=action_spec,
    observation_spec=observation_spec,
    horizon=1000,
    gamma=0.99,
    n_envs=100,
    use_mjwarp=True,   # switch MJX to the MjWarp backend
    nconmax=3000,      # optional: max number of contacts for warp
    njmax=None,        # optional: max number of constraints for warp
)
```

- **From high-level factories** (e.g. `ImitationFactory`), you can forward the same flag:

```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make(
    "MjxUnitreeG1",
    default_dataset_conf=dict(task="stepinplace1"),
    use_mjwarp=True,
)
```

For a complete example, see `examples/tutorials/02_creating_mjx_env.py`.

---

## Real Robot Deployment

<div align="center">
  <img width="80%" src="https://raw.githubusercontent.com/robfiras/loco-mujoco-real/main/assets/x2_sim2real.gif" alt="The AgiBot X2 Ultra running a tracking policy, on the robot and in simulation"/>
</div>

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) is the companion repository for taking
policies trained in LocoMuJoCo to a real robot. It currently covers the **Unitree G1** and the **AgiBot X2 Ultra**
and provides:

- **deployable training setups**: for each robot an environment with the observations, rewards and sim-to-real
  randomization it needs, and ready-to-run training configs for motion tracking with **PPO** and with **SuperMimic**,
- **deployment tools**: export of a trained policy to a bundle that runs with only `numpy` and `mujoco`, validation
  of the bundle against the training environment, rollouts in plain MuJoCo, sim-to-sim over the robot's middleware,
  and the controller that runs on the robot.

```bash
python training/<robot>/train.py             # PPO
python training/<robot>/train_supermimic.py  # SuperMimic
```

The video shows a SuperMimic tracking policy on the AgiBot X2 Ultra, on the robot and in simulation. The configs for
training such a policy are in that repository, for the X2
([`training/x2/supermimic.yaml`](https://github.com/robfiras/loco-mujoco-real/blob/main/training/x2/supermimic.yaml))
and for the G1 ([`training/g1/supermimic.yaml`](https://github.com/robfiras/loco-mujoco-real/blob/main/training/g1/supermimic.yaml)).

The [real robot deployment tutorial](https://loco-mujoco.readthedocs.io/en/latest/source/tutorials/real_robot_deployment.html)
explains the stages from training to the robot.

---

## Environments 
You want a quick overview of all **environments** available? You can find it 
[here](/loco_mujoco/environments) and more detailed in the [Documentation](https://loco-mujoco.readthedocs.io/).

<div align="center">
  <img src="imgs/lmj_envs.gif"/>
</div>

And stay tuned! There are many more to come ...

---

## Tutorials

We provide a set of tutorials to help you get started with LocoMuJoCo. You can find them in the [tutorials folder](./examples/tutorials)
or with more explanation in the [documentation](https://loco-mujoco.readthedocs.io/).

If you want to check out training examples of a PPO, GAIL, AMP, or DeepMimic agent, you can find them 
in the [training examples folder](./examples/training_examples). For instance, [here](./examples/training_examples/jax_rl_mimic) is an example of a DeepMimic agent
you can train to achieve a human-like walking in all directions, which was trained in 36 min on an RTX 3080 Ti:

<div align="center">
  <img src="imgs/unitree_h1_walk_anydir.gif"/>
</div>

---
## Citation
```
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```




