Basics
=================================

Task-IDs
-----------------------------------
Tasks are chosen by defining Task-IDs. The general structure of a Task-Id is `<environment>.<task>.<dataset_type>`.
For the Task-ID, you have to choose *at least* the environment name. Missing information will be filled with default setting
specific to the respective environment. Default settings can be seen in the :code:`generate()` method of the
respective environment class. A list of all available *Task-IDs* in LocoMuJoCo is given below.
Alternatively, you can use the following code:

.. code-block:: python

    from loco_mujoco import LocoEnv

    task_ids = LocoEnv.get_all_task_names()

Some environments have additional information in the Task-ID. For example, the HumanoidTorque4Ages and
HumanoidMuscle4Ages environments have different `modes` for different ages, where an index is added to the task
name to chose which of the 4 Humanoids to choose. The general structure there is
`<environment>.<task>.<mode>.<dataset_type>`. The smallest humanoid is indexed by 1 while the adult is indexed
by 4; e.g. :code:`"HumanoidMuscle4Ages.walk.1.real"` or :code:`"HumanoidMuscle4Ages.run.4.real"`.

Given a Task-ID, it is very straightforward to create an environment. For example, for a MushrooRL environment:


.. code-block:: python

    from loco_mujoco import LocoEnv

    # create an environment with a Muscle Humanoid running with motion capture data (real dataset type)
    env = LocoEnv("HumanoidMuscle.run.real")


You can do the same for a Gymnasium environment:

.. code-block:: python

    import loco_mujoco  # needed to register the environments
    import gym

    # create an environment with a Muscle Humanoid running with motion capture data (real dataset type)
    env = gym.make("LocoMujoco", env_name="HumanoidMuscle.run.real")

Replay Datasets
-----------------------------------

If you would like to visualize the datasets, you can replay them using the following code:

.. literalinclude:: ../../examples/replay_datasets/replay_humanoid_muscle.py
    :language: python

This method will read the joint positions in the dataset and set the joint positions of the humanoid accordingly
at each time step. Alternatively, you can also replay the dataset from the velocities. To do so, the first state of a
trajectory is sampled and all following states will be calculated from velocities using numerical integration.
This method is useful to verify that the velocities in the dataset are consistent with the positions. Here is an example:

.. literalinclude:: ../../examples/replay_datasets/replay_talos.py
    :language: python

Both, the :code:`play_trajectory` and :code:`play_trajectory_from_velocity` methods are available
for the Gymnasium interface as well.

.. note:: Dynamics are disregard when replaying a dataset!

.. _env-label:
Overview of Environments, Tasks and Datasets
-----------------------------------

Here you can find an overview of all tasks. For each task, you can choose between different datasets. The real dataset
contains real (noisy) motion capture datasets without actions. The perfect dataset contains ground truth states and
actions form an expert policy. The preference dataset contains preferences of an ground truth expert with states and
action (only available on an few tasks). The status of a dataset can be seen down below. ✅ means it is already
available, and 🔶 means pending.

.. note::
    The **perfect** and **preference** datasets are only available for the *default* settings of an environment.
    By default, arms are not included in the observation space. Hence, these datasets are only available
    for tasks without arms. In comparison, the real dataset also contains arm observations.

.. list-table::
   :widths: 25 30 15 30
   :header-rows: 1

   * - **Image**
     - **Task-IDs**
     - **Status Datasets**
     - **Description**
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/cdcd4617-c18a-448d-b42a-ea01384016b0
     - | HumanoidMuscle.walk
       | HumanoidMuscle.run
       | HumanoidMuscle4Ages.walk.4
       | HumanoidMuscle.run
       | HumanoidMuscle4Ages.run.4
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  Task of an adult **Muscle** Humanoid
        Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/1bdebeb1-401c-439e-8f2e-7197fd34c5e5
     - | HumanoidMuscle4Ages.walk.3
       | HumanoidMuscle4Ages.run.3
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~12 year old) **Muscle** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/2c7aeb58-65a6-427c-8b12-197c96410cd8
     - | HumanoidMuscle4Ages.walk.2
       | HumanoidMuscle4Ages.run.2
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~5-6 year old) **Muscle** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/600a917f-c784-4b5e-ac99-9472711de843
     - | HumanoidMuscle4Ages.walk.1
       | HumanoidMuscle4Ages.run.1
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~2 year old) **Muscle** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/cf32520f-5a2e-401a-9f2e-b8033ef7109c
     - | HumanoidMuscle.walk
       | HumanoidMuscle.run
       | HumanoidMuscle4Ages.walk.4
       | HumanoidMuscle.run
       | HumanoidMuscle4Ages.run.4
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  Task of an adult **Torque** Humanoid
        Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/352f3594-8903-4eaf-a223-f751b590f4ec
     - | HumanoidMuscle4Ages.walk.3
       | HumanoidMuscle4Ages.run.3
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~12 year old) **Torque** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/06c83af9-3c45-43a1-8173-aa1d8771fe4c
     - | HumanoidMuscle4Ages.walk.2
       | HumanoidMuscle4Ages.run.2
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~5-6 year old) **Torque** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/5ec93baa-bed8-4d9f-b983-3bc12264b9b6
     - | HumanoidMuscle4Ages.walk.1
       | HumanoidMuscle4Ages.run.1
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Task of a (~2 year old) **Torque** Humanoid Walking or Running Straight.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/fed0315c-921e-4b2e-a9c2-54b85198ef65
     - | UnitreeH1.walk
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  UnitreeH1 Straight Walking Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/ab0dec59-fc24-4763-8ff6-38d58ac3b3de
     - | UnitreeH1.run
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  UnitreeH1 Straight Running Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/851ff3c0-d05f-4de1-a00b-7b3204056e2f
     - | UnitreeH1.carry
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  UnitreeH1 Straight Carry Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/22c7bb0c-ff92-4e99-a964-7654df6d22c4
     - | Talos.walk
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  Talos Straight Walking Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/0ba1f0e7-1f3d-4088-a44f-0a53bec1cf3a
     - | Talos.carry
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Talos Straight Carry Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/1ff09d98-e46b-429c-ac07-87de58853d28
     - | Atlas.walk
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  Atlas Straight Walking Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/3c433333-3466-445b-b39f-2f990553d5ff
     - | Atlas.carry
     - | real: ✅
       | perfect: 🔶
       | preference: 🔶
     -  Atlas Straight Carry Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/b722bb42-a26c-4692-b1a8-c6f71a78e37b
     - | UnitreeA1.simple
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  UnitreeA1 Straight Walking Task.
   * - .. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/5a1f783e-8b52-4680-a22b-d96f89faf9b3
     - | UnitreeA1.hard
     - | real: ✅
       | perfect: ✅
       | preference: 🔶
     -  UnitreeA1 Walking in **8 Different Direction** Task.