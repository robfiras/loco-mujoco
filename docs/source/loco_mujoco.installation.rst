Getting Started
================

Installation
--------------


Clone the repo and install in editable mode:

.. code-block:: bash

   cd loco-mujoco
   pip install -e .

To use **JAX with GPU**:

.. code-block:: bash

   pip install jax["cuda12"]

.. note::

   To run the **MyoSkeleton** environment, run:

   ``loco-mujoco-myomodel-init`` to accept the license and download the model.

Datasets
--------------

LocoMuJoCo provides three sources of mocap data: **default**, **LAFAN1**, and **AMASS**.
The default and LAFAN1 datasets are downloaded automatically from the `HuggingFace dataset repository <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets>`_.

AMASS must be installed manually due to licensing—see the :ref:`installation instructions <amass_installation>`.


Example usage:

.. code-block:: python

   from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf

   env, traj = ImitationFactory.make(
       "UnitreeH1",
       default_dataset_conf=DefaultDatasetConf(["squat"]),
       lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
       # amass_dataset_conf=AMASSDatasetConf([...])
   )

   env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)

Speeding Up Dataset Loading
------------------------------

LocoMuJoCo only stores joint positions and velocities in its datasets. All other attributes are computed with forward
kinematics upon loading. To accelerate loading, you can cache the results:

.. code-block:: bash

   loco-mujoco-set-all-caches --path <path to cache>

Example:

.. code-block:: bash

   loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"

Using MJX with the MjWarp Backend
-----------------------------------

MJX environments can run on the standard MJX backend or on the **MjWarp** backend, which can further accelerate
parallel simulation on compatible GPUs. Pass ``use_mjwarp=True`` (and optionally ``nconmax`` and ``njmax``) when creating the
environment:

.. code-block:: python

   from loco_mujoco import ImitationFactory

   env, traj = ImitationFactory.make(
       "MjxUnitreeG1",
       default_dataset_conf=dict(task="stepinplace1"),
       use_mjwarp=True,
   )

Environments
--------------

An overview of all environments is available `here <https://github.com/robfiras/loco-mujoco/tree/dev/loco_mujoco/environments>`_
and in more detail in the `documentation <https://loco-mujoco.readthedocs.io/>`_.

.. image:: https://github.com/user-attachments/assets/bf5eb128-eedc-49f7-a64c-cef0072d53f3
   :align: center

Stay tuned — more coming soon!

Tutorials
--------------

Find tutorials in the `tutorials folder <https://github.com/robfiras/loco-mujoco/tree/dev/examples/tutorials>`_
or explore the full tutorials in the `online documentation <https://loco-mujoco.readthedocs.io/>`_.

Citation
--------------

If you use LocoMuJoCo in your research, please cite:

.. code-block:: bibtex

   @inproceedings{alhafez2023b,
     title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
     author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
     booktitle={6th Robot Learning Workshop, NeurIPS},
     year={2023}
   }
