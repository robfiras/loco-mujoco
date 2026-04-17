Creating New Environments
=========================

.. todo::

    A full step-by-step tutorial on how to define, register, and publish a new
    humanoid/quadruped environment will be added here soon. For now, the
    section below covers how datasets are handled for a new environment.


Datasets for New Environments
------------------------------

When you add a **custom environment** that is not on the
`LocoMuJoCo HuggingFace hub <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main>`_
(e.g. a new robot, a variant of an existing one, or something registered from a downstream package),
none of the published DefaultDataset / Lafan1Dataset clips will match your env's skeleton. LocoMuJoCo
handles this transparently: whenever :class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`
fails to find a dataset for your env on the hub, it falls back to **retargeting the clip on-the-fly**
from a known source env into your env's skeleton, and caches the result on disk. Subsequent calls
reuse the cached file.

Prerequisites
^^^^^^^^^^^^^^

1. **SMPL robot configuration YAML** — a retargeting config must exist for your env (same format as the
   built-in confs under ``loco_mujoco/smpl/robot_confs/``). Drop the YAML into a directory of your choice
   and register that directory:

   .. code-block:: bash

       loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH --value /path/to/my/robot_confs

2. **SMPL body model path** — set once with:

   .. code-block:: bash

       loco-mujoco-set-smpl-model-path --path /path/to/smpl/models

3. *(optional)* **Custom MuJoCo models path** — if your env XML lives outside the main repo:

   .. code-block:: bash

       loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_MODELS_PATH --value /path/to/my/models

How it is triggered
^^^^^^^^^^^^^^^^^^^^

The fallback runs automatically on the first call that references a missing dataset:

.. code-block:: python

    from loco_mujoco import ImitationFactory

    # MyCustomHumanoid is not on the hub → retarget on-the-fly from the source env,
    # then cache under LOCOMUJOCO_CONVERTED_DEFAULT_PATH/mocap/MyCustomHumanoid/squat.npz
    env, traj = ImitationFactory.make(
        "MyCustomHumanoid",
        default_dataset_conf=dict(task="squat"),
    )

The source env used for retargeting is read from the target env's info properties and defaults to:

- :meth:`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid.default_dataset_source_env` → ``"SkeletonTorque"``
- :meth:`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid.lafan1_dataset_source_env` → ``"UnitreeH1v2"``

Override them on your env class if a different source gives better retargeting quality (e.g. an
env resembling the Unitree G1 should use ``UnitreeG1`` as its LAFAN1 source):

.. code-block:: python

    from loco_mujoco.core.utils import info_property
    from loco_mujoco.environments.humanoids import BaseRobotHumanoid

    class MyUnitreeG1Variant(BaseRobotHumanoid):
        @info_property
        def lafan1_dataset_source_env(self) -> str:
            return "UnitreeG1"

Pre-retargeting everything ahead of time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On-the-fly retargeting can be slow on the first call (several minutes per clip) and competes with
JAX for GPU memory. For a one-time bulk retargeting of every published clip for your new env, use:

.. code-block:: bash

    loco-mujoco-retarget-for-new-env --env MyCustomHumanoid --import-module my_package

``--import-module`` loads the Python package(s) that register your environment (so
``MyCustomHumanoid`` is visible to the registry). By default both DefaultDataset and Lafan1Dataset
clips are retargeted, skipping clips that already exist in the cache. Useful flags:

- ``--dataset-types default lafan1`` — choose which sources to process (default: both).
- ``--overwrite`` — re-retarget even if the cache already contains the clip.

The script sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` up-front so that JAX does not grab 75% of
the GPU (which would starve the PyTorch-based SMPL fitting).
