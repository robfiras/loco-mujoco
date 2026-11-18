# API reference

The API reference is generated from the docstrings of the `loco_mujoco` package. Each subpackage below
has its own page listing its modules, classes and functions.

For explanations and examples, start with the [user guide](../user_guide/index.md).

| Package | Contents |
|---|---|
| {mod}`loco_mujoco.core` | Simulation base classes ({class}`~loco_mujoco.core.mujoco_base.Mujoco`, {class}`~loco_mujoco.core.mujoco_mjx.Mjx`) and the pluggable modules: observations, rewards, control functions, domain randomization, terrain, initial and terminal state handlers, trajectories, wrappers and visuals. |
| {mod}`loco_mujoco.environments` | {class}`~loco_mujoco.environments.base.LocoEnv` and all humanoid and quadruped environments. |
| {mod}`loco_mujoco.task_factories` | {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`, {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` and the dataset configurations. |
| {mod}`loco_mujoco.algorithms` | JAX implementations of PPO, GAIL, AMP, SuperMimic and the experimental algorithms. |
| {mod}`loco_mujoco.smpl` | SMPL-based human-to-robot and robot-to-robot retargeting. |
| {mod}`loco_mujoco.utils` | Trajectory metrics, dataset path helpers, logging and video utilities. |

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:
   :caption: Packages

   loco_mujoco.core
   loco_mujoco.environments
   loco_mujoco.task_factories
   loco_mujoco.algorithms
   loco_mujoco.smpl
   loco_mujoco.utils
```
