"""
Dataset configurations for :class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`.
"""

from typing import Union, ClassVar, List
from dataclasses import dataclass

from loco_mujoco.core.trajectory.dataclasses import Trajectory


@dataclass
class DefaultDatasetConf:
    """
    Configuration for loading default datasets provided by LocoMuJoCo.

    Attributes:
        task (Union[str, list]): The task or list of tasks to load, e.g. ``"walk"``.
        dataset_type (str): The type of the dataset to load. Only "mocap" (motion capture data) is available.
        debug (bool): Debug flag. Currently not used when loading the dataset.

    """

    task: Union[str, list]  = "walk"
    dataset_type: str = "mocap"
    debug: bool = False

    def __post_init__(self):
        if self.dataset_type != "mocap":
            raise ValueError(f"Unknown dataset type: '{self.dataset_type}'. Only 'mocap' is available.")


@dataclass
class AMASSDatasetConf:
    """
    Configuration for loading AMASS datasets.

    At least one of the two attributes has to be set; if both are set, the datasets of both are loaded.

    Attributes:
        rel_dataset_path (Union[str, list]): A relative path or a list of relative paths to
            load from the AMASS dataset, relative to the AMASS directory and without file extension.
        dataset_group (str): A name of a predefined group of datasets to load from AMASS. Available:
            ``"AMASS_LOCOMOTION_DATASETS"``.

    """
    rel_dataset_path: Union[str, list] = None
    dataset_group: str = None

    def __post_init__(self):
        assert self.rel_dataset_path is not None or self.dataset_group is not None, ("Either `rel_dataset_path` or "
                                                                                     "`dataset_group` must be set.")


@dataclass
class LAFAN1DatasetConf:
    """
    Configuration for loading LAFAN1 datasets.

    At least one of the two attributes has to be set; if ``dataset_group`` is set, ``dataset_name`` is ignored.

    Attributes:
        dataset_name (Union[str, list]): A name of a dataset or a list of dataset names to load from LAFAN1.
        dataset_group (str): A name of a predefined group of datasets to load from LAFAN1. Available:
            ``"LAFAN1_LOCOMOTION_DATASETS"``, ``"LAFAN1_DANCE_DATASETS"`` and ``"LAFAN1_ALL_DATASETS"``.

    .. note:: This dataset is loaded from LocoMuJoCo's HuggingFace repository:
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets. For environments that have no dataset
        there, it is retargeted from another environment.

    """

    dataset_name: Union[str, list] = None
    dataset_group: str = None

    def __post_init__(self):
        assert self.dataset_name is not None or self.dataset_group is not None, ("Either `dataset_name` or "
                                                                                 "`dataset_group` must be set.")


@dataclass
class CustomDatasetConf:
    """
    Configuration for loading custom trajectories.

    Attributes:
        traj (Trajectory): A custom trajectory to load.

    """
    traj: Trajectory
