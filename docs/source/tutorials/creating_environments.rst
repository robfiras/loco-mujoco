Creating Environments
=======================

This section provides examples of creating environments in LocoMuJoCo. We cover three different environment setups: a standard MuJoCo environment, an MJX environment in JAX, and a MuJoCo environment integrated with Gymnasium.

Task Factories
--------------

LocoMuJoCo comes with two task factories, :class:`ImitationFactory <loco_mujoco.task_factories.imitation_factory.ImitationFactory>` and :class:`RLFactory <loco_mujoco.task_factories.rl_factory.RLFactory>`, and a :class:`GymnasiumWrapper <loco_mujoco.core.wrappers.gymnasium.GymnasiumWrapper>` built on top of them.

Both factories return a tuple of the environment and its trajectory, ``env, traj = ImitationFactory.make(...)``. The trajectory is ``None`` for the :class:`RLFactory <loco_mujoco.task_factories.rl_factory.RLFactory>`.

ImitationFactory
^^^^^^^^^^^^^^^^^

The :class:`ImitationFactory <loco_mujoco.task_factories.imitation_factory.ImitationFactory>` is used for imitation learning tasks. It sets up the environment with a dataset and predefines the following components:

- **Initial State:** The default initial state is :class:`TrajInitialStateHandler <loco_mujoco.core.initial_state_handler.traj_init_state.TrajInitialStateHandler>`, which initializes the environment from a state in the trajectory.
- **Terminal State:** The default terminal state is :class:`RootPoseTrajTerminalStateHandler <loco_mujoco.core.terminal_state_handler.traj.RootPoseTrajTerminalStateHandler>`, which terminates the environment when the root pose reaches a threshold determined from the trajectory.

Any of the default components can be overridden by passing the desired component as an argument to the factory.

The imitation factory requires at least one trajectory to be defined. Trajectories can be loaded from different sources such as:

- **Default Datasets:** Configured using :class:`DefaultDatasetConf <loco_mujoco.task_factories.dataset_confs.DefaultDatasetConf>`, these datasets are predefined and include common tasks like "walk" or "stepinplace1". The dataset type can be either "mocap" or "pretrained".
- **AMASS Datasets:** Configured using :class:`AMASSDatasetConf <loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf>`, these datasets are typically motion capture data from the AMASS database. You can specify a relative path to the dataset or a predefined group of datasets.
- **LAFAN1 Datasets:** Configured using :class:`LAFAN1DatasetConf <loco_mujoco.task_factories.dataset_confs.LAFAN1DatasetConf>`, these datasets are loaded from the LocoMuJoCo's HuggingFace repository: `robfiras/loco-mujoco-datasets <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets>`_. They provide datasets for all humanoid environments.
- **Custom Trajectories:** Configured using :class:`CustomDatasetConf <loco_mujoco.task_factories.dataset_confs.CustomDatasetConf>`, these datasets allow loading user-defined trajectories, which must be provided as a :class:`~loco_mujoco.core.trajectory.dataclasses.Trajectory` object.

Check out the :ref:`Replay Datasets Tutorial <replay_datasets_tutorial>` to see how these datasets can be loaded.

RLFactory
^^^^^^^^^^^^^^^^^

The :class:`RLFactory <loco_mujoco.task_factories.rl_factory.RLFactory>` is used for reinforcement learning tasks. It sets up the environment with a robot and predefines the following components:

- **Initial State:** The default initial state is :class:`DefaultInitialStateHandler <loco_mujoco.core.initial_state_handler.default.DefaultInitialStateHandler>`, which initializes the environment with the default state.
- **Terminal State:** The default terminal state is :class:`HeightBasedTerminalStateHandler <loco_mujoco.core.terminal_state_handler.height.HeightBasedTerminalStateHandler>`, which terminates the environment when the robot falls below a certain height.
- **Goal:** The default goal is :class:`GoalRandomRootVelocity <loco_mujoco.core.observations.goals.GoalRandomRootVelocity>`, which sets a random goal for the robot's root velocity.
- **Reward:** The default reward is :class:`TargetVelocityGoalReward <loco_mujoco.core.reward.default.TargetVelocityGoalReward>`, which rewards the robot for reaching the goal velocity.

Similar to the imitation factory, the default components can be overridden by providing the desired component as an argument to the factory.

GymnasiumWrapper
^^^^^^^^^^^^^^^^^

We provide a Gymnasium wrapper to integrate MuJoCo environments with Gymnasium. We do not support MJX environments with Gymnasium yet. Depending on whether a dataset configuration is provided or not, the wrapper will create an imitation or RL environment with the default components defined above.

Creating a MuJoCo Environment
-------------------------------

In this example, we create a MuJoCo environment using the :class:`ImitationFactory <loco_mujoco.task_factories.imitation_factory.ImitationFactory>`.
We use the MyoSkeleton model, which requires running ``loco-mujoco-myomodel-init`` first, and load the ``stepinplace1`` dataset. After creating the environment, the script initializes it, performs random actions, and renders the simulation. The loop runs for 1000 steps or until the environment signals that it has reached an absorbing state.

.. literalinclude:: ../../../examples/tutorials/01_creating_mujoco_env.py
   :language: python
   :linenos:

Note that we could also use the :class:`RLFactory <loco_mujoco.task_factories.rl_factory.RLFactory>` to create the environment. In that case, we would not need to provide a dataset configuration. This also holds for the MJX example.

Creating an MJX Environment
---------------------------

This example demonstrates creating an MJX environment using the :class:`ImitationFactory <loco_mujoco.task_factories.imitation_factory.ImitationFactory>` with JAX. The environment MjxUnitreeG1 is created with a ``squat`` task. The script leverages JAX's `vmap` to efficiently handle parallel environments and actions. The rendering is also performed in parallel, and the script periodically logs the speed of environment steps. Note that the dynamics of all rendered robots are independent of each other.

.. literalinclude:: ../../../examples/tutorials/02_creating_mjx_env.py
   :language: python
   :linenos:

Creating a MuJoCo Environment with Gymnasium
----------------------------------------------

This example shows how to integrate a MuJoCo environment with :class:`GymnasiumWrapper <loco_mujoco.core.wrappers.gymnasium.GymnasiumWrapper>`. Here, we use the SkeletonTorque model. The environment is configured to use the default ``walk`` and the LAFAN1 ``walk1_subject1`` datasets and to render in human mode. The script initializes the environment, performs random actions, and visualizes the simulation.

.. literalinclude:: ../../../examples/tutorials/04_creating_gymansium_env.py
   :language: python
   :linenos:
