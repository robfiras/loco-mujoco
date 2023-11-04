# LocoMuJoCo

<p align="center">
  <img src="https://media.giphy.com/media/5N0bdxFKv8c640WMC6/giphy-downsized-large.gif" />
</p>

**LocoMuJoCo** is an **imitation learning benchmark** specifically targeted towards **locomotion**. It encompasses a diverse set of environments, including quadrupeds, bipeds, and musculoskeletal human models, each accompanied by comprehensive datasets, such as real noisy motion capture data, ground truth expert data, and ground truth sub-optimal data,
enabling evaluation across a spectrum of difficulty levels. 

### Key Advatanages 
✅ Easy to use with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) or [Mushroom-RL](https://github.com/MushroomRL/mushroom-rl) interface \
✅ Many environments including humanoids and quadrupeds \
✅ Diverse set of datasets --> e.g., noisy motion capture or ground truth datasets with actions \
✅ Wide spectrum spectrum of difficulty levels \
✅ Built-in domain randomization \
✅ Many baseline algorithms for quick benchmarking 

---

## Installation

To install this repository, clone it and then run:

```bash
cd loco-mujoco
pip install -e . 
```

To verify that everything is installed correctly, run the examples such as:

```bash
python examples/simple_mushroom_env/example_unitree_a1.py
```

To replay a dataset run:

```bash
python examples/replay_datasets/replay_Unitree.py
```


---
## Example
LocoMuJoCo is very easy to use. Just choose and create the environment, and generate the dataset belonging to this task and you are ready to go! 
```python
import numpy as np
import loco_mujoco
import gymnasium as gym


env = gym.make("LocoMujoco", env_name="HumanoidTorque.run")
dataset = env.create_dataset()
```

---
## Citation
```
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
```

