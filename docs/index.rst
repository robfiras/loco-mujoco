.. LocoMuJoCo documentation master file, created by
   sphinx-quickstart on Tue Jan  9 19:32:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LocoMuJoCo!
======================================

.. image::  https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a
   :width: 70%
   :align: center


**LocoMuJoCo** is an **imitation learning benchmark** specifically targeted towards **locomotion**. It encompasses a diverse set of environments, including quadrupeds, bipeds, and musculoskeletal human models, each accompanied by comprehensive datasets, such as real noisy motion capture data, ground truth expert data, and ground truth sub-optimal data,
enabling evaluation across a spectrum of difficulty levels.

**LocoMuJoCo** also allows you to specify your own reward function to use this benchmark for **pure reinforcement learning**! Checkout the example below!

.. figure:: https://github.com/robfiras/loco-mujoco/assets/69359729/c16dfa4a-4fdb-4701-9a42-54cbf7644301
   :align: center

|

The **core idea** behind LocoMuJoCo is to allow researcher focussing on imitation or reinforce learning to transition from simple
toy task in locomotion to realistic and complex environments crucial for real-world applications. At the same time, we wanted to LocoMuJoCo
to be as simple to use as possible by providing comprehensive datasets for each environment and task in a **single line of code**!

Key Advantages
----------------

| ✅ Easy to use with `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ or `Mushroom-Rl <https://github.com/MushroomRL/mushroom-rl>`_ interface \
| ✅ Many environments including humanoids and quadrupeds \
| ✅ Diverse set of datasets --> e.g., noisy motion capture or ground truth datasets with actions \
| ✅ Wide spectrum spectrum of difficulty levels \
| ✅ Built-in domain randomization \
| ✅ Many baseline algorithms for quick benchmarking


.. toctree::
   :caption: Documentation
   :maxdepth: 3
   :hidden:

   source/loco_mujoco.installation.rst
   source/loco_mujoco.api.rst


.. toctree::
   :caption: Tutorials
   :hidden:

   source/tutorials/interfaces.rst
   source/tutorials/imitation_learning.rst
   source/tutorials/reinforcement_learning.rst
   source/tutorials/domain_randomization.rst


