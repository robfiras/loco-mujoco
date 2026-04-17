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
.. image:: https://readthedocs.org/projects/loco-mujoco/badge/?version=latest
   :target: https://loco-mujoco.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/pypi/v/loco-mujoco
   :target: https://pypi.org/project/loco-mujoco/
.. image:: https://img.shields.io/badge/Discord-Join%20Us-7289DA?style=flat&logo=discord&logoColor=white
   :target: https://discord.gg/gEqR3xCVdn

Latest News
===========

🚀 **Since Release: v1.0**

LocoMuJoCo now supports MJX and includes new JAX algorithms, expanded environments, and over **22,000 datasets**!

Overview
========

**LocoMuJoCo** is an **imitation learning benchmark** tailored for **whole-body control**.
It includes a diverse range of environments—**quadrupeds**, **humanoids**, and **(musculo-)skeletal human models**—each equipped with comprehensive datasets (22k+ samples per humanoid).

While designed for imitation learning, it also supports **pure reinforcement learning** with custom reward classes.

.. image:: ../imgs/main_lmj.gif
   :align: center

Key Advantages
--------------

- ✅ Supports **MuJoCo** (single) and **MJX** (parallel) environments
- ✅ Includes **12 humanoid + 4 quadruped environments**, with **4 biomechanical human models**
- ✅ Clean, single-file **JAX algorithms**: PPO, GAIL, AMP, DeepMimic
- ✅ **22,000+ motion capture datasets** (AMASS, LAFAN1, native)
- ✅ **Robot-to-robot retargeting**
- ✅ Trajectory comparison metrics (e.g., DTW, Fréchet distance) implemented in **JAX**
- ✅ **Gymnasium interface**
- ✅ Built-in **domain and terrain randomization**
- ✅ **Modular design**: easily swap components (observations, rewards, terminal handlers, etc.)
- ✅ Comprehensive [documentation](https://loco-mujoco.readthedocs.io/)

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
   source/tutorials/trajectory_interface.rst



