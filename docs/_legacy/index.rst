.. LocoMuJoCo documentation master file, created by
   sphinx-quickstart on Tue Jan  9 19:32:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LocoMuJoCo!
======================================

.. image:: https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a
   :width: 70%
   :align: center

.. image:: https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev
.. image:: https://app.readthedocs.org/projects/loco-mujoco/badge/?version=latest
   :target: https://loco-mujoco.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

Latest News
===========

🚀 **New in Release v1.2**

LocoMuJoCo now includes **SuperMimic** and supports **real-robot deployment** on the **AgiBot X2 Ultra** and the **Unitree G1** through
`LocoMuJoCo Real <https://github.com/robfiras/loco-mujoco-real>`__.

Overview
========

**LocoMuJoCo** is an **imitation learning benchmark** tailored for **whole-body control**.
It includes a diverse range of environments—**quadrupeds**, **humanoids**, and **(musculo-)skeletal human models**—each equipped with comprehensive datasets.

While designed for imitation learning, it also supports **pure reinforcement learning** with custom reward classes.

.. image:: ../imgs/main_lmj.gif
   :align: center

Key Advantages
--------------

- ✅ Supports **MuJoCo** (single) and **MJX/MjWarp** (parallel) environments
- ✅ Includes **12 humanoid + 4 quadruped environments**, with **4 biomechanical human models**
- ✅ Clean, single-file **JAX algorithms**: SuperMimic, PPO, GAIL, AMP, DeepMimic
- ✅ **22,000+ motion capture datasets** (AMASS, LAFAN1, native)
- ✅ **Robot-to-robot retargeting**
- ✅ Trajectory comparison metrics (e.g., DTW, Fréchet distance) implemented in **JAX**
- ✅ **Gymnasium interface**
- ✅ Built-in **domain and terrain randomization**
- ✅ **Real-robot deployment** with `LocoMuJoCo Real <https://github.com/robfiras/loco-mujoco-real>`__: deployable
  training setups for PPO and SuperMimic, and the tools to take a policy to the robot (:ref:`tutorial <real_robot_deployment_tutorial>`)
- ✅ **Modular design**: easily swap components (observations, rewards, terminal handlers, etc.)
- ✅ Comprehensive `documentation <https://loco-mujoco.readthedocs.io/>`__

.. toctree::
   :caption: Documentation
   :maxdepth: 3
   :hidden:

   source/loco_mujoco.installation.rst
   source/loco_mujoco.amass_installation.rst
   source/loco_mujoco.api.rst


.. toctree::
   :caption: Tutorials
   :hidden:

   source/tutorials/replay_datasets.rst
   source/tutorials/creating_environments.rst
   source/tutorials/customizing_environments.rst
   source/tutorials/creating_new_environments.rst
   source/tutorials/real_robot_deployment.rst
   source/tutorials/trajectory_interface.rst
   source/tutorials/rendering.rst



