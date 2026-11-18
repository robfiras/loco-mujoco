---
html_theme.sidebar_secondary.remove: true
---

# LocoMuJoCo

<div class="lmj-hero">
  <img class="lmj-hero-logo dark-light" src="_static/logo.png" alt="LocoMuJoCo">
  <p class="lmj-tagline">Imitation and reinforcement learning for whole-body control,
  from motion capture data to the real robot.</p>
  <p class="lmj-sub">16 humanoids, quadrupeds and musculoskeletal human models. Retargeted motion capture
  datasets. JAX algorithms that train in minutes on one GPU. One environment API for MuJoCo, MJX and MuJoCo Warp.</p>
  <div class="lmj-hero-buttons">
    <a class="sd-btn sd-btn-primary" href="getting_started/index.html">Get started</a>
    <a class="sd-btn sd-btn-outline-secondary" href="environments/index.html">Browse environments</a>
    <a class="sd-btn sd-btn-outline-secondary" href="https://github.com/robfiras/loco-mujoco">GitHub</a>
  </div>
</div>

```{image} ../imgs/main_lmj.gif
:alt: Humanoid robots and human models of LocoMuJoCo replaying motion capture data
:align: center
:width: 100%
```

<div class="lmj-stats">
  <div class="lmj-stat"><span class="lmj-stat-num">16</span><span class="lmj-stat-label">embodiments</span></div>
  <div class="lmj-stat"><span class="lmj-stat-num">3</span><span class="lmj-stat-label">simulation backends</span></div>
  <div class="lmj-stat"><span class="lmj-stat-num">10</span><span class="lmj-stat-label">JAX algorithms</span></div>
  <div class="lmj-stat"><span class="lmj-stat-num">3</span><span class="lmj-stat-label">motion capture sources</span></div>
  <div class="lmj-stat"><span class="lmj-stat-num">2</span><span class="lmj-stat-label">robots deployed on hardware</span></div>
</div>

## Install

```bash
git clone https://github.com/robfiras/loco-mujoco.git
cd loco-mujoco
pip install -e .
```

Then replay a motion capture clip on a humanoid robot:

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

env, traj = ImitationFactory.make("UnitreeG1", default_dataset_conf=DefaultDatasetConf(["walk"]))
env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

The [installation guide](getting_started/installation.md) covers GPU support and the optional extras.

## Where to go next

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.3em` Get started
:link: getting_started/index
:link-type: doc

Install LocoMuJoCo, create your first environment, replay a motion and train a tracking policy.
:::

:::{grid-item-card} {octicon}`book;1.3em` User guide
:link: user_guide/index
:link-type: doc

How environments, trajectories, datasets and the pluggable modules fit together.
:::

:::{grid-item-card} {octicon}`cpu;1.3em` Training
:link: training/index
:link-type: doc

PPO, DeepMimic-style tracking, AMP, GAIL, SuperMimic and more, with ready-to-run configs.
:::

:::{grid-item-card} {octicon}`zap;1.3em` Sim-to-real
:link: sim2real/index
:link-type: doc

Take a trained policy to the Unitree G1 or the AgiBot X2 Ultra with LocoMuJoCo Real.
:::
::::

## What is in the box

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} One API, three backends
:link: user_guide/backends
:link-type: doc

The same environment runs on MuJoCo (CPU), on MJX and on MuJoCo Warp (GPU, thousands of parallel
environments).
:::

:::{grid-item-card} 16 embodiments
:link: environments/index
:link-type: doc

Nine humanoid robots, three (musculo-)skeletal human models and four quadrupeds.
:::

:::{grid-item-card} Motion capture data for every humanoid
:link: user_guide/datasets
:link-type: doc

A native dataset, LAFAN1 and AMASS, retargeted to each humanoid and downloaded on demand.
:::

:::{grid-item-card} Retargeting
:link: user_guide/retargeting
:link-type: doc

Human-to-robot retargeting through SMPL and robot-to-robot retargeting of any existing dataset.
:::

:::{grid-item-card} Single-file JAX algorithms
:link: training/index
:link-type: doc

Environment and learner are compiled into one function, so a tracking policy trains in minutes.
:::

:::{grid-item-card} Pluggable modules
:link: user_guide/modular_architecture
:link-type: doc

Observations, rewards, control, domain randomization, terrain and termination are interchangeable
components.
:::

:::{grid-item-card} Trajectory metrics
:link: training/evaluation
:link-type: doc

Euclidean distance, dynamic time warping and discrete Fréchet distance, written in JAX.
:::

:::{grid-item-card} Domain and terrain randomization
:link: user_guide/domain_randomization
:link-type: doc

Randomize the model, the state, the observations and the actions, and train on rough terrain.
:::

:::{grid-item-card} Real-robot deployment
:link: sim2real/index
:link-type: doc

Deployable training setups and the tools to run a policy on the robot.
:::
::::

## Real-robot deployment

```{image} https://raw.githubusercontent.com/robfiras/loco-mujoco-real/main/assets/x2_sim2real.gif
:alt: The AgiBot X2 Ultra running a SuperMimic tracking policy, on the robot and in simulation
:align: center
:width: 80%
```

<p class="lmj-caption">A SuperMimic tracking policy on the AgiBot X2 Ultra, on the robot and in simulation.</p>

[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) is the companion repository for taking
policies trained in LocoMuJoCo to hardware. See the [sim-to-real guide](sim2real/index.md).

## Citing LocoMuJoCo

If you use LocoMuJoCo in your research, please cite it. The BibTeX entry is on the
[citation page](development/citation.md).

```{toctree}
:hidden:
:maxdepth: 2

Get started <getting_started/index>
User guide <user_guide/index>
Training <training/index>
Environments <environments/index>
Sim-to-real <sim2real/index>
API <api/index>
Development <development/index>
```
