API Documentation
====================


Environments
-----------

.. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/73ca0cdd-3958-4d59-a1f7-0eba00fe373a


LocoMuJoCo focuses on *Loco*motion environments. This includes humanoids and quadrupeds, with
a strong focus on the latter. The environment is built on top of MuJoCo. The aim of LocoMuJoCo is to be
a simple and easy-to-use environment for imitation learning and reinforcement learning while shifting the focus towards
realistic and complex tasks crucial for real-world robotics. LocoMuJoCo strives to be simple and user-friendly, offering
an environment tailored for both imitation and reinforcement learning. Its main objective is to shift the focus away from
simplistic locomotion tasks often used as benchmarks for imitation and reinforcement learning algorithms, and instead
prioritize realistic and intricate tasks vital for real-world robotics applications.

For imitation learning, it is crucial to have a good and diverse datasets. LocoMuJoCo makes it very simple to generate
diverse datasets of different difficulty levels in a single line of code. This allows the user to focus on the learning
algorithm and not worry about the environment. Here is a simple example of how to generate a the environment and the dataset
for the Unitree H1 robot:

.. literalinclude:: ../../examples/simple_gymnasium_env/example_unitree_h1.py
    :language: python


.. toctree::
    :hidden:

    ./loco_mujoco.environments.rst
    ./loco_mujoco.environments.humanoids.rst
    ./loco_mujoco.environments.quadrupeds.rst
    ./loco_mujoco.environments.base.rst


Datasets
-----------


Rewards
-----------
.. toctree::
    :hidden:

    ./loco_mujoco.rewards.rst
