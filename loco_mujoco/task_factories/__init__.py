"""
Task factories, the main entry point for creating environments.

:class:`~loco_mujoco.task_factories.rl_factory.RLFactory` creates plain reinforcement learning tasks,
:class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` creates imitation learning tasks from
dataset configurations (default datasets, AMASS, LAFAN1 or custom trajectories). Importing this package registers
both factories.
"""

from .base import TaskFactory
from .rl_factory import RLFactory
from .imitation_factory import ImitationFactory
from .dataset_confs import DefaultDatasetConf, AMASSDatasetConf, LAFAN1DatasetConf, CustomDatasetConf


# register factories
RLFactory.register()
ImitationFactory.register()


