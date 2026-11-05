# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocoMuJoCo is an imitation learning benchmark for whole-body locomotion control, supporting 12 humanoid and 4 quadruped environments with motion capture datasets (AMASS, LAFAN1). It supports both CPU simulation (MuJoCo) and GPU-accelerated parallel simulation (MJX/MjWarp via JAX).

## Commands

**Install:**
```bash
pip install -e .
# GPU JAX support (optional):
pip install jax["cuda12"]
# SMPL/AMASS support (optional):
loco-mujoco-myomodel-init
```

**Run tests:**
```bash
pytest --ignore=tests/test_task_factories.py
# Single test file:
pytest tests/test_reward.py
```

**CLI utilities (dataset management):**
```bash
loco-mujoco-download           # Download all datasets
loco-mujoco-download-real      # Download real mocap datasets
loco-mujoco-download-perfect   # Download perfect/synthetic datasets
loco-mujoco-set-amass-path     # Configure AMASS path
loco-mujoco-set-lafan1-path    # Configure LAFAN1 path
```

## Architecture

### Simulation backends (`loco_mujoco/core/`)
- `mujoco_base.py` — CPU simulation base class wrapping raw MuJoCo
- `mujoco_mjx.py` — GPU simulation extending base with JAX/vmap/jit support
- `observations/`, `reward/`, `control_functions/`, `domain_randomizer/`, `initial_state_handler/`, `terminal_state_handler/`, `terrain/`, `visuals/`, `wrappers/` — pluggable modular components

### Environments (`loco_mujoco/environments/`)
- `base.py` — `LocoEnv` class extending `MujocoMjx`; adds trajectory following, imitation learning, and dataset integration. This is the main base class all environments inherit from.
- Humanoid and quadruped subclasses live in subdirectories here.

### Task factories (`loco_mujoco/task_factories/`)
- `ImitationFactory` — creates imitation learning tasks with dataset configs
- `RLFactory` — creates plain RL tasks
- Main entry point for users creating environments programmatically

### Algorithms (`loco_mujoco/algorithms/`)
- JAX/Flax implementations of PPO, GAIL, AMP, DeepMimic
- Common utilities: `networks.py`, `dataclasses.py`, `base_algorithm.py`

### Trajectories & datasets (`loco_mujoco/core/trajectory/`, `loco_mujoco/datasets/`)
- `TrajectoryHandler` manages motion capture data loading and sampling
- Trajectory metrics: DTW, discrete Fréchet distance
- Dataset classes for AMASS, LAFAN1, and native datasets

### Key design patterns
- **Modular plug-in architecture**: reward functions, observations, control types, domain randomization are all interchangeable components passed at environment construction.
- **JAX-first**: Flax dataclasses, `vmap`, `jit` used throughout for performance. Stateful sim wrapped in `SimulationState` dataclasses for functional programming compatibility.
- **Dual backend**: same environment API works on both CPU (MuJoCo) and GPU (MJX); the backend is selected at construction time.

### Public API
`loco_mujoco/__init__.py` exports the top-level API. `loco_mujoco/core/README.md` and `loco_mujoco/environments/README.md` contain usage examples and environment status tables.

## Notes
- CI runs on pushes to `master` and `dev` branches using Python 3.11.
- `tests/test_task_factories.py` is excluded from CI runs.
- Hydra is used for configuration management in training examples (`examples/training_examples/`).
