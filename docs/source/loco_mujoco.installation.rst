Getting Started
====================

Installation
----------------
You have the choice to install the latest release via PyPI by running

.. code-block:: bash

    pip install loco-mujoco


or you do an editable installation by cloning this repository and then running:

.. code-block:: bash

    git clone --recurse-submodules git@github.com:robfiras/loco-mujoco.git
    cd loco-mujoco
    pip install -e .

.. note::
        We fixed the version of MuJoCo to 2.3.7 during installation since we found that there are slight
        differences in the simulation, which made testing very difficult. However, in practice, you can
        use any newer version of MuJoCo! Just install it after installing LocoMuJoCo.

.. note::
        If you want to run the **MyoSkeleton** environment, you need to additionally run
        ``loco-mujoco-myomodel-init`` to accept the license and download the model. Finally, you need to
        upgrade Mujoco to 3.2.2 and dm_control to 1.0.22 *after* installing this package and downloading the datasets!

Download the Datasets
---------------------

After installing LocoMuJoCo, new commands for downloading the datasets will be setup for you.
You have the choice of downloading all datasets available or only the ones you need.
For example, run the following command to install all datasets:

.. code-block:: bash

    loco-mujoco-download


Run the following command to install only the real (motion capture, no actions) datasets:

.. code-block:: bash

    loco-mujoco-download-real


Run the following command to install only the perfect (ground-truth with actions) datasets:

.. code-block:: bash

    loco-mujoco-download-perfect

.. _install-baseline-label:
Installing the Baselines
-----------------------
If you also want to run the baselines, you have to install our imitation learning library `imitation_lib <https://github.com/robfiras/ls-iq>`__.


Verify Installation
-------------

To verify that everything is installed correctly, run the examples such as:

.. code-block:: bash

    python examples/simple_mushroom_env/example_unitree_a1.py


To replay a dataset run:

.. code-block:: bash

    python examples/replay_datasets/replay_Unitree.py


Environments & Tasks
---------------------

You want a quick overview of all **environments**, **tasks** and **datasets** available?
:doc:`Here <loco_mujoco.environments>` you can find it.

.. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/73ca0cdd-3958-4d59-a1f7-0eba00fe373a
    :align: center

And stay tuned! There are many more to come ...


Quick Examples
---------------------

LocoMuJoCo is very easy to use. Just choose and create the environment, and generate the dataset belonging to this task and you are ready to go!

.. code-block:: python

    import numpy as np
    import loco_mujoco
    import gymnasium as gym


    env = gym.make("LocoMujoco", env_name="HumanoidTorque.run")
    dataset = env.create_dataset()

You want to use LocoMuJoCo for pure reinforcement learning? No problem! Just define your custom reward function and pass it to the environment!

.. code-block:: python

    import numpy as np
    import loco_mujoco
    import gymnasium as gym
    import numpy as np


    def my_reward_function(state, action, next_state):
        return -np.mean(action)


    env = gym.make("LocoMujoco", env_name="HumanoidTorque.run", reward_type="custom",
                   reward_params=dict(reward_callback=my_reward_function))



LocoMuJoCo *natively* supports `MushroomRL <https://github.com/MushroomRL/mushroom-rl>`__:

.. code-block:: python

    import numpy as np
    from loco_mujoco import LocoEnv

    env = LocoEnv.make("HumanoidTorque.run")
    dataset = env.create_dataset()


You can find many more examples `here <https://github.com/robfiras/loco-mujoco/tree/master/examples>`__.


Citation
---------------------

.. code-block::

    @inproceedings{alhafez2023b,
    title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
    author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
    booktitle={6th Robot Learning Workshop, NeurIPS},
    year={2023}
    }

Credits
---------------------
Both Unitree models were taken from the `MuJoCo menagerie <https://github.com/google-deepmind/mujoco_menagerie>`__.
