.. _replay_datasets_tutorial:

Replaying Datasets
=====================================

This example shows how to use the `ImitationFactory` from `loco_mujoco` to load and play back datasets
on a Unitree H1 robot model, though any other environment works accordingly.
You can easily specify different datasets, such as:

- `DefaultDataset <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main>`__ : Collected by the maintainers of LocoMuJoCo.
- `Lafan1Dataset <https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main>`__ : Humanoid motion capture dataset.
- `AMASSDataset <https://amass.is.tue.mpg.de/>`__ : Humanoid motion capture dataset.

The environment supports loading multiple datasets across formats, adjusting the substeps, and visualizing
the trajectory through the built-in `play_trajectory` method.

.. literalinclude:: ../../../examples/tutorials/00_replay_datasets.py
   :language: python
   :linenos:
