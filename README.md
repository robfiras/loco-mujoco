<p align="center">
  <img width="70%" src="https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a">
</p>

![continous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/loco-mujoco/badge/?version=latest)](https://loco-mujoco.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/loco-mujoco)](https://pypi.org/project/loco-mujoco/)

**LocoMuJoCo** is an **imitation learning benchmark** specifically targeted towards **locomotion**. It encompasses a diverse set of environments, including quadrupeds, bipeds, and musculoskeletal human models, each accompanied by comprehensive datasets, such as real noisy motion capture data, ground truth expert data, and ground truth sub-optimal data,
enabling evaluation across a spectrum of difficulty levels. 

**LocoMuJoCo** also allows you to specify your own reward function to use this benchmark for **pure reinforcement learning**! Checkout the example below!

<p align="center">
  <img src="https://github.com/robfiras/loco-mujoco/assets/69359729/c16dfa4a-4fdb-4701-9a42-54cbf7644301">
</p>

### Key Advantages 
✅ Easy to use with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) or [Mushroom-RL](https://github.com/MushroomRL/mushroom-rl) interface \
✅ Many environments including humanoids and quadrupeds \
✅ Diverse set of datasets --> e.g., noisy motion capture or ground truth datasets with actions \
✅ Wide spectrum spectrum of difficulty levels \
✅ Built-in domain randomization \
✅ Many baseline algorithms for quick benchmarking \
✅ [Documentation](https://loco-mujoco.readthedocs.io/)

---

## Installation

You have the choice to install the latest release via PyPI by running 

```bash
pip install loco-mujoco 
```

or you do an editable installation by cloning this repository and then running:

```bash
cd loco-mujoco
pip install -e . 
```

> [!NOTE]
> We fixed the version of MuJoCo to 2.3.7 during installation since we found that there are slight 
> differences in the simulation, which made testing very difficult. However, in practice, you can 
> use any newer version of MuJoCo! Just install it after installing LocoMuJoCo.

### Download the Datasets
After installing LocoMuJoCo, new commands for downloading the datasets will be setup for you.
You have the choice of downloading all datasets available or only the ones you need.

For example, to install all datasets run:  
```bash
loco-mujoco-download
```

To install only the real (motion capture, no actions) datasets run:  
```bash
loco-mujoco-download-real
```

To install only the perfect (ground-truth with actions) datasets run:  
```bash
loco-mujoco-download-perfect
```

### Installing the Baselines
If you also want to run the baselines, you have to install our imitation learning library [imitation_lib](https://github.com/robfiras/ls-iq). You find example files for training the baselines for any LocoMuJoCo task [here](examples/imitation_learning).

### First Test
To verify that everything is installed correctly, run the examples such as:

```bash
python examples/simple_mushroom_env/example_unitree_a1.py
```

To replay a dataset run:

```bash
python examples/replay_datasets/replay_Unitree.py
```
---
## Environments & Tasks
You want a quick overview of all **environments**, **tasks** and **datasets** available? You can find it 
[here](/loco_mujoco/environments) and more detailed in the [Documentation](https://loco-mujoco.readthedocs.io/).

<p align="center">
  <img src="https://github.com/robfiras/loco-mujoco/assets/69359729/73ca0cdd-3958-4d59-a1f7-0eba00fe373a">
</p>

And stay tuned! There are many more to come ...

---
## Quick Examples
LocoMuJoCo is very easy to use. Just choose and create the environment, and generate the dataset belonging to this task and you are ready to go! 
```python
import numpy as np
import loco_mujoco
import gymnasium as gym


env = gym.make("LocoMujoco", env_name="HumanoidTorque.run")
dataset = env.create_dataset()
```
You want to use LocoMuJoCo for pure reinforcement learning? No problem! Just define your custom reward function and pass it to the environment!

```python
import numpy as np
import loco_mujoco
import gymnasium as gym
import numpy as np


def my_reward_function(state, action, next_state):
    return -np.mean(action)


env = gym.make("LocoMujoco", env_name="HumanoidTorque.run", reward_type="custom",
               reward_params=dict(reward_callback=my_reward_function))

```

LocoMuJoCo *natively* supports [MushroomRL](https://github.com/MushroomRL/mushroom-rl):

```python
import numpy as np
from loco_mujoco import LocoEnv

env = LocoEnv.make("HumanoidTorque.run")
dataset = env.create_dataset()
```

You can find many more examples [here](examples).

Detailed Tutorials are given in the [Documentation](https://loco-mujoco.readthedocs.io/).

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

---
## Credits 
Both Unitree models were taken from the [MuJoCo menagerie](https://github.com/google-deepmind/mujoco_menagerie)



