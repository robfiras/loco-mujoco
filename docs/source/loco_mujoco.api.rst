API Documentation
====================

Core
-----------

The core consists of fundamental modules that serve as the backbone for all tasks and environments.
It supports both reinforcement learning and imitation learning, providing essential components such
as the main simulation in MuJoCo and Mjx, observation and goal interfaces, initial and terminal state
handlers, control and reward functions, as well as domain and terrain interfaces.

.. toctree::
    :hidden:

    ./loco_mujoco.core.rst


Environments
----------------

LocoMuJoCo comes with 16 different environments, including 12 humanoids and 4 quadrupeds. The environments
are designed to be simple and user-friendly, offering a variety of tasks for both imitation and reinforcement
learning. The environments are built on top of the core modules, providing a seamless integration with the
rest of the library.


.. toctree::
    :hidden:

    ./loco_mujoco.environments.humanoids.rst
    ./loco_mujoco.environments.quadrupeds.rst



Datasets
-----------

LocoMuJoCo provides different datasets for imitation learning. For now, three different dataset sources are available:

- **DefaultDataset**: Collected by the maintainers of LocoMuJoCo.
- **Lafan1Dataset**: Humanoid motion capture dataset.
- `AMASSDataset <https://amass.is.tue.mpg.de/>`__ : Humanoid motion capture dataset.

All datasets will be automatically downloaded, retargeted for the respective humanoid and optionally cached.
Due to licensing, the AMASS dataset needs to be downloaded manually. You can find more information on the installation
page of the documentation. The DefaultDataset and Lafan1Dataset are hosted on
`LocoMuJoCo's HuggingFace hub <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main>`__.


Trajectory Format
-------------------

The trajectory module defines a structured format for representing and manipulating humanoid motion
trajectories in MuJoCo-based environments. It provides a modular, serializable, and extensible data pipeline suitable
for imitation learning, motion analysis, and reinforcement learning tasks.

At the core is the :class:`~loco_mujoco.trajectory.dataclasses.Trajectory` class, which bundles together all components of a trajectory,
including static model and scene information (:class:`~loco_mujoco.trajectory.dataclasses.TrajectoryInfo`), time-varying motion data
(:class:`~loco_mujoco.trajectory.dataclasses.TrajectoryData`), optional RL-compatible transitions (:class:`~loco_mujoco.trajectory.dataclasses.TrajectoryTransitions`),
and an observation container used to reconstruct high-level observations.

- :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryInfo` holds metadata and structural information about the MuJoCo model,
  such as joint types, joint names, and body/site mappings, as well as the data recording frequency.
- :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryModel` (used within :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryInfo`) captures a reduced snapshot
  of the MuJoCo model state, enabling model-aware operations such as joint/body/site additions, removals, and reordering.
- :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryData` is a batchable and memory-efficient representation of motion data, including
  joint positions (:attr:`qpos`), velocities (:attr:`qvel`), body positions and orientations (:attr:`xpos`, :attr:`xquat`), contact forces,
  and other physical quantities. It supports dynamic slicing and interpolation for resampling trajectories to new frequencies.
- :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryTransitions` contains tuples of observations, actions, rewards, absorbing flags,
  and done flags, structured for reinforcement learning pipelines. This component is optional and primarily used for turning demonstration
  data into agent-ready training transitions.
- :class:`~loco_mujoco.trajectory.dataclasses.SingleData` is a lightweight, single-frame version of :class:`~loco_mujoco.trajectory.dataclasses.TrajectoryData`,
  used to extract and operate on individual time steps of motion.

Together, these components support advanced trajectory manipulation, serialization to `.npz` format, and backend-agnostic operation
(NumPy or JAX).

.. toctree::
    :hidden:

    ./loco_mujoco.trajectory.rst


Task Factories
----------------

For easy dataset loading, LocoMuJoCo provides task factories that can be used to load the datasets and create the
corresponding tasks. Different sources of datasets can be mixed and matched to create a single task. Here is an example
of an imitation learning task using the DefaultDataset and Lafan1Dataset.

.. toctree::
    :hidden:

    ./loco_mujoco.task_factories.rst


.. literalinclude:: ../../examples/replay_datasets/example.py
    :language: python





