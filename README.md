<p align="center">
  <img width="70%" src="https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a">
</p>

![continuous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![Documentation Status](https://app.readthedocs.org/projects/loco-mujoco/badge/?version=latest)](https://loco-mujoco.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/loco-mujoco)](https://pypi.org/project/loco-mujoco/)
[![Python](https://img.shields.io/pypi/pyversions/loco-mujoco)](https://pypi.org/project/loco-mujoco/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Latest News:**
> A **new release (v1.2)** just dropped! 🎉  
> **New:** LocoMuJoCo now includes **SuperMimic** and supports **real-robot deployment** on the
> **AgiBot X2 Ultra** and the **Unitree G1** through
> [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real).

**LocoMuJoCo** is an **imitation learning benchmark** specifically designed for **whole-body control**.
It features a diverse set of environments, including **quadrupeds**, **humanoids**, and **(musculo-)skeletal human models**,
each provided with comprehensive datasets. Although primarily focused on imitation learning, it also supports
custom reward functions, making it suitable for pure reinforcement learning as well.

<div align="center">
  <img src="imgs/main_lmj.gif"/>
</div>

**[Documentation](https://loco-mujoco.readthedocs.io/)** ·
**[Get started](https://loco-mujoco.readthedocs.io/en/latest/getting_started/index.html)** ·
**[Environments](https://loco-mujoco.readthedocs.io/en/latest/environments/index.html)** ·
**[Training](https://loco-mujoco.readthedocs.io/en/latest/training/index.html)** ·
**[Sim-to-real](https://loco-mujoco.readthedocs.io/en/latest/sim2real/index.html)** ·
**[API](https://loco-mujoco.readthedocs.io/en/latest/api/index.html)**

### Key Advantages
✅ Supports **MuJoCo** (single environment) and **MJX/MJWarp** (parallel environments) \
✅ Includes **12 humanoid and 4 quadruped environments**, featuring 3 **biomechanical human models** \
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

LocoMuJoCo requires **Python 3.10 or newer**. We recommend a fresh environment, for example with conda:

```bash
conda create -n loco-mujoco python=3.11
conda activate loco-mujoco
```

Install the latest release from PyPI:

```bash
pip install loco-mujoco
```

Or, to run the examples, change the code or use the SMPL/AMASS extras, clone this repo and do an editable
installation:

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
pip install -e .
```

By default, this installs the CPU version of JAX. To use JAX on the GPU:

```bash
pip install "jax[cuda12]<=0.10.1,!=0.10.0"
```

The robot models (about 200 MB) are downloaded automatically the first time you create an environment.
The [installation guide](https://loco-mujoco.readthedocs.io/en/latest/getting_started/installation.html) covers
the optional extras, offline use, the MyoSkeleton model and the
[AMASS setup](https://loco-mujoco.readthedocs.io/en/latest/getting_started/amass.html).

## Quick start

Replay motion capture data on a humanoid robot:

```python
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf

env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=DefaultDatasetConf(["squat"]),
                                  lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]))

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

Train a policy that tracks a motion with PPO and a DeepMimic-style reward on thousands of parallel MJX environments:

```bash
cd examples/training_examples/jax_rl_mimic
python experiment.py
```

<div align="center">
  <img src="imgs/unitree_h1_walk_anydir.gif"/>
</div>

The [getting-started guide](https://loco-mujoco.readthedocs.io/en/latest/getting_started/index.html) walks through
these steps, and the [tutorials folder](./examples/tutorials) and the
[training examples folder](./examples/training_examples) contain runnable scripts for every feature and algorithm.

---

## Environments

LocoMuJoCo includes nine humanoid robots, three (musculo-)skeletal human models and four quadrupeds. The
[environment gallery](https://loco-mujoco.readthedocs.io/en/latest/environments/index.html) lists them with their
observation and action spaces.

<div align="center">
  <img src="imgs/lmj_envs.gif"/>
</div>

---

## Real Robot Deployment

<div align="center">
  <img width="80%" src="https://raw.githubusercontent.com/robfiras/loco-mujoco-real/main/assets/x2_sim2real.gif" alt="The AgiBot X2 Ultra running a tracking policy, on the robot and in simulation"/>
</div>

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) is the companion repository for taking
policies trained in LocoMuJoCo to a real robot. It currently covers the **Unitree G1** and the **AgiBot X2 Ultra**
and provides deployable training setups for **PPO** and **SuperMimic** and the tools to export, validate and run a
policy on the robot. The video shows a SuperMimic tracking policy on the AgiBot X2 Ultra, on the robot and in
simulation. The [sim-to-real guide](https://loco-mujoco.readthedocs.io/en/latest/sim2real/index.html) explains the
stages from training to the robot.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report issues, set up a development
environment and open a pull request, and [CHANGELOG.md](CHANGELOG.md) for the release history.

## Citation

If you use LocoMuJoCo in your research, please cite the paper on the current version:

```bibtex
@article{alhafez2026locomujoco,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Whole-Body Control},
author={Firas Al-Hafez and Davide Tateo and Oleg Kaidanov and Aditya Bhatt and Artur Zelik and Zhengyi Luo and Jan Peters},
journal={arXiv preprint arXiv:2611.26293},
year={2026}
}
```

and the original paper, which describes version 0.1.0 ([arXiv:2311.02496](https://arxiv.org/abs/2311.02496)):

```bibtex
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```
