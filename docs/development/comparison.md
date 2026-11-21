# Comparison to related software

Several open-source libraries address learning for humanoid robots and simulated characters. This page
places LocoMuJoCo among them, so that you can judge which tool fits your work. It is written by the
LocoMuJoCo authors, and we have tried to describe the other projects fairly.

```{important}
The comparison was compiled in 2026 for the release of LocoMuJoCo 1.2.0 (November 2026), from the cited
papers and the public documentation of each project. All of these projects are under active development
and their features change. Check the linked project pages for the current state, and please
[open an issue](https://github.com/robfiras/loco-mujoco/issues) if an entry is out of date.
```

## Scope of each project

| Project | Main focus | Simulation |
|---|---|---|
| [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) | Benchmark of reinforcement learning tasks for humanoids, covering locomotion and manipulation | MuJoCo on the CPU |
| [MyoSuite](https://github.com/MyoHub/myosuite) | Musculoskeletal models and motor-control tasks | MuJoCo on the CPU |
| [mjlab](https://github.com/mujocolab/mjlab) | Robot learning framework with the manager-based environment interface of Isaac Lab | MuJoCo Warp on the GPU |
| [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | Framework for humanoid simulation and motion tracking at scale | Isaac Gym, Isaac Lab, Genesis, Newton |
| [MimicKit](https://github.com/xbpeng/MimicKit) | Reinforcement learning framework for motion imitation methods | GPU simulators; not JAX-based |
| **LocoMuJoCo** | Imitation learning benchmark for whole-body control with datasets, retargeting, algorithms and metrics | MuJoCo on the CPU, MJX and MuJoCo Warp on the GPU |

## Feature comparison

The table uses five properties that distinguish the libraries for imitation learning on humanoids. A check
mark means that the feature is built in; a dash means that it is not part of the project according to the
sources named above. A dash is not a judgement: most of these projects do not aim to provide the feature.

| | One API for a NumPy and a JAX backend | Built-in retargeting | Muscle-actuated models | Trajectory-similarity metrics | Sim-to-real tooling |
|---|:---:|:---:|:---:|:---:|:---:|
| HumanoidBench | – | – | – | – | – |
| MyoSuite | – | – | ✓ | – | – |
| mjlab | – | – | – | – | ✓ |
| ProtoMotions | – | ✓ | – | – | ✓ |
| MimicKit | – | – | – | – | – |
| **LocoMuJoCo** | ✓ | ✓ | ✓ | ✓ | ✓ |

What the columns mean for LocoMuJoCo:

- **One API for a NumPy and a JAX backend.** Every environment runs serially on MuJoCo with NumPy and in
  parallel on MJX or MuJoCo Warp with JAX, and the environment components are written once for both. See
  [simulation backends](../user_guide/backends.md).
- **Built-in retargeting.** Human motion capture data is retargeted to any registered humanoid, and
  trajectories can be transferred from one robot to another. See [retargeting](../user_guide/retargeting.md).
- **Muscle-actuated models.** The skeleton models can be driven by muscles as well as by torques.
- **Trajectory-similarity metrics.** Euclidean distance, dynamic time warping and discrete Fréchet distance
  between rollouts and reference motion. See [evaluation](../training/evaluation.md).
- **Sim-to-real tooling.** The companion repository
  [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) exports, validates and deploys policies
  on the Unitree G1 and the AgiBot X2 Ultra. See [sim-to-real](../sim2real/index.md).

## The projects in detail

### HumanoidBench

[HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)
([Sferrazza et al., 2024](https://doi.org/10.15607/RSS.2024.XX.061)) is a benchmark of 27 whole-body
locomotion and manipulation tasks, built mainly around the Unitree H1 with dexterous hands. Its
contribution is the task suite, which covers long-horizon and contact-rich behaviour that LocoMuJoCo does
not address. It targets reinforcement learning: it ships no expert demonstrations or motion capture data,
and it runs on CPU MuJoCo without a JAX backend. Choose HumanoidBench to evaluate reinforcement learning
algorithms on diverse humanoid tasks, in particular with manipulation; choose LocoMuJoCo when the task is
defined by reference motion.

### MyoSuite

[MyoSuite](https://github.com/MyoHub/myosuite)
([Caggiano et al., 2022](https://arxiv.org/abs/2205.13600)) provides physiologically detailed
musculoskeletal models of the human body, including a muscle-actuated full-body skeleton, with
motor-control tasks for reinforcement learning. It is a reference for biomechanical fidelity. LocoMuJoCo's
`MyoSkeleton` environment uses the [MyoSkeleton model](https://github.com/myolab/myo_model) published by
MyoLab. MyoSuite does not provide motion capture datasets for
imitation or a retargeting pipeline, and its focus is not sim-to-real transfer to robots. Choose MyoSuite
for biomechanics and motor-control research on detailed muscle models; LocoMuJoCo offers muscle-actuated
humanoids within the same imitation learning setup as its robots.

### mjlab

[mjlab](https://github.com/mujocolab/mjlab)
([Zakka et al., 2026](https://arxiv.org/abs/2601.22074)) combines the manager-based environment interface
of Isaac Lab with MuJoCo Warp for GPU simulation and trains with
[RSL-RL](https://github.com/leggedrobotics/rsl_rl). It ships reference tasks for velocity tracking, motion
imitation and manipulation and is positioned for sim-to-real reinforcement learning. It runs on the GPU
only, and it does not include a set of retargeted motion capture datasets, cross-robot retargeting or
biomechanical models. The environment design of LocoMuJoCo Real follows mjlab's G1 motion-tracking setup
closely. Choose mjlab if you work in the Isaac Lab style with PyTorch; LocoMuJoCo is built on JAX, with
training and simulation compiled into one function.

### ProtoMotions

[ProtoMotions](https://github.com/NVlabs/ProtoMotions) (Tessler et al., 2025) is a GPU-accelerated
framework for humanoid simulation and motion tracking that supports several simulators and includes
human-to-robot retargeting based on [PyRoki](https://arxiv.org/abs/2505.03728). It demonstrates transfer of
a tracking policy to the Unitree G1. It is a
toolkit more than a benchmark: it does not define a fixed dataset suite with an evaluation protocol based
on trajectory-similarity metrics. Choose ProtoMotions for motion tracking on the simulators it supports;
LocoMuJoCo stays within the MuJoCo family and adds a CPU backend, datasets per robot and metrics.

### MimicKit

[MimicKit](https://github.com/xbpeng/MimicKit)
([Peng, 2025](https://arxiv.org/abs/2510.13794)) collects motion imitation methods, among them DeepMimic,
AMP and ADD, in one modular reinforcement learning framework aimed at physics-based character animation.
It is a methods toolkit and largely complementary to a benchmark: it does not ship a suite of robots with
datasets, biomechanical models, trajectory metrics or sim-to-real tooling. Choose MimicKit for reference
implementations of imitation methods from the character-animation literature; LocoMuJoCo implements a
smaller set of such methods (DeepMimic-style tracking, AMP, GAIL) on its own environments.

## Retargeting pipelines

LocoMuJoCo's retargeting builds on the SMPL-fitting approach of
[PHC](https://github.com/ZhengyiLuo/PHC) (Luo et al., 2023) and
[H2O](https://github.com/LeCAR-Lab/human2humanoid) (He et al., 2024): the shape of the SMPL body model is
fitted to the robot once, and motion is then transferred by tracking corresponding points. Those
implementations target individual robots. LocoMuJoCo turns the approach into a pipeline driven by one
configuration file per robot: a new robot needs the configuration and 15 named sites in its model, the
resulting targets are tracked with soft constraints in MuJoCo, and the same mapping is reused to
transfer trajectories from one robot to another. This pipeline was introduced in version 1.0.1.

Later projects take different routes:

- [GMR](https://github.com/YanjieZe/GMR) ([Araújo et al., 2026](https://arxiv.org/abs/2510.02252)) replaces
  the shape fit by non-uniform scaling and differential inverse kinematics. It targets real-time use and
  fewer retargeting artifacts.
- [OmniRetarget](https://arxiv.org/abs/2509.26633) (Yang et al., 2025) preserves the interaction with
  objects and terrain during retargeting, which LocoMuJoCo's pipeline does not model.

If retargeting quality for a particular robot or interaction-rich motion is your main concern, these
tools are worth evaluating; trajectories they produce can be loaded into LocoMuJoCo as
[custom trajectories](../user_guide/custom_trajectories.md).

## What LocoMuJoCo does not cover

To complete the picture:

- LocoMuJoCo has no suite of manipulation or object-interaction tasks. Its tasks are locomotion with a
  velocity command and tracking of reference motion.
- It supports the MuJoCo family of simulators only (MuJoCo, MJX, MuJoCo Warp), not Isaac Lab, Genesis or
  Newton.
- The algorithms are written in JAX. Users of PyTorch can use the environments through the
  [Gymnasium interface](../user_guide/gymnasium.md) on the CPU, but the included training code does not
  carry over.

## Software LocoMuJoCo builds on

LocoMuJoCo depends on the [MuJoCo](https://github.com/google-deepmind/mujoco) physics engine with MJX and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), the SMPL body model, the AMASS and LAFAN1
motion capture datasets, and [MetrX](https://github.com/pompetzki/metrx) for the trajectory metrics.
The [citation page](citation.md) lists the references.
