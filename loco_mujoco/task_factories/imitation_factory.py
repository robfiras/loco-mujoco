"""
Factory for imitation learning tasks.
"""

import os
from typing import Union, Dict, Tuple
from omegaconf import ListConfig, DictConfig
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
import yaml

import loco_mujoco
from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.core.trajectory import Trajectory, TrajectoryHandler
from loco_mujoco.smpl.retargeting import (load_retargeted_amass_trajectory, extend_motion,
                                          load_robot_conf_file, retarget_traj_from_robot_to_robot)
from loco_mujoco.smpl.const import AMASS_LOCOMOTION_DATASETS
from loco_mujoco.datasets.humanoids.LAFAN1 import load_lafan1_trajectory
from loco_mujoco.datasets.humanoids.LAFAN1 import (LAFAN1_LOCOMOTION_DATASETS,
                                                   LAFAN1_DANCE_DATASETS, LAFAN1_ALL_DATASETS)


# Default source envs used when retargeting on-the-fly. Can be overridden per-env
# via the `default_dataset_source_env` / `lafan1_dataset_source_env` info properties
# on the env class (see BaseRobotHumanoid).
_DEFAULT_RETARGET_SOURCE_ENV = "SkeletonTorque"
_LAFAN1_RETARGET_SOURCE_ENV = "UnitreeH1v2"

from .base import TaskFactory
from .dataset_confs import DefaultDatasetConf, AMASSDatasetConf, LAFAN1DatasetConf, CustomDatasetConf


class ImitationFactory(TaskFactory):
    """
    A factory class for creating imitation learning environments with arbitrary trajectories.

    Use :meth:`make` to create an environment and load the trajectories of the given dataset configurations. The
    trajectories can come from the default datasets of LocoMuJoCo, from AMASS, from LAFAN1 and from custom
    trajectories, and the sources can be combined.

    """

    @classmethod
    def make(cls, env_name: str,
             default_dataset_conf: Union[DefaultDatasetConf, Dict, DictConfig] = None,
             amass_dataset_conf: Union[AMASSDatasetConf, Dict, DictConfig] = None,
             lafan1_dataset_conf: Union[LAFAN1DatasetConf, Dict, DictConfig] = None,
             custom_dataset_conf: Union[CustomDatasetConf, Dict, DictConfig] = None,
             terminal_state_type: str = "RootPoseTrajTerminalStateHandler",
             init_state_type: str = "TrajInitialStateHandler",
             **kwargs) -> Tuple[LocoEnv, Trajectory]:
        """
        Create and return an imitation learning environment given different dataset configurations.

        The trajectories of all given configurations are loaded, concatenated and passed to the environment. Each
        configuration can be given as the dataclass or as a dictionary with its fields. At least one configuration is
        needed.

        Args:
            env_name (str): The name of the registered environment to create.
            default_dataset_conf (DefaultDatasetConf, optional): The configuration for the default trajectory.
            amass_dataset_conf (AMASSDatasetConf, optional): The configuration for the AMASS trajectory.
            lafan1_dataset_conf (LAFAN1DatasetConf, optional): The configuration for the LAFAN1 trajectory.
            custom_dataset_conf (CustomDatasetConf, optional): The configuration for a custom trajectory.
            terminal_state_type (str, optional): The terminal state handler to use.
                Defaults to "RootPoseTrajTerminalStateHandler".
            init_state_type (str, optional): The initial state handler to use. Defaults to "TrajInitialStateHandler".
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            Tuple[LocoEnv, Trajectory]: The requested imitation learning environment with the trajectory
            preloaded, and the processed trajectory.

        Raises:
            KeyError: If the specified environment is not registered.
            ValueError: If a dataset type or dataset group is unknown.

        """

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered LocoMuJoCo environment.")

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Create and return the environment
        env = env_cls(init_state_type=init_state_type, terminal_state_type=terminal_state_type, **kwargs)

        all_trajs = []

        # Load the default trajectory if available
        if default_dataset_conf is not None:
            if isinstance(default_dataset_conf, (dict, DictConfig)):
                default_dataset_conf = DefaultDatasetConf(**default_dataset_conf)
            all_trajs.append(cls.get_default_traj(env, default_dataset_conf))

        # Load the AMASS trajectory if available
        if amass_dataset_conf is not None:
            if isinstance(amass_dataset_conf, (dict, DictConfig)):
                amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)
            all_trajs.append(cls.get_amass_traj(env, amass_dataset_conf))

        # Load the LAFAN1 trajectory if available
        if lafan1_dataset_conf is not None:
            if isinstance(lafan1_dataset_conf, (dict, DictConfig)):
                lafan1_dataset_conf = LAFAN1DatasetConf(**lafan1_dataset_conf)
            all_trajs.append(cls.get_lafan1_traj(env, lafan1_dataset_conf))

        # Load the custom trajectory if available
        if custom_dataset_conf is not None:
            if isinstance(custom_dataset_conf, (dict, DictConfig)):
                custom_dataset_conf = CustomDatasetConf(**custom_dataset_conf)
            all_trajs.append(cls.get_custom_dataset(env, custom_dataset_conf))

        # concatenate trajectories
        all_trajs = Trajectory.concatenate(all_trajs)

        # add to the environment
        traj = env.process_trajectory(all_trajs)

        return env, traj

    @staticmethod
    def get_default_traj(env, default_dataset_conf) -> Trajectory:
        """
        Load the default trajectories of the tasks in the configuration.

        For each task, the trajectory is taken from the local cache (``LOCOMUJOCO_CONVERTED_DEFAULT_PATH``) if it exists
        there. Otherwise it is downloaded from the HuggingFace dataset repository or, if the repository has no dataset
        for this environment, retargeted from the dataset of another environment; the result is completed with the
        model-specific quantities and written to the cache. The trajectories are then processed to match the model and
        the control frequency of the environment and concatenated.

        Args:
            env: The environment for which the trajectories are loaded.
            default_dataset_conf (DefaultDatasetConf): The configuration for the default trajectory.

        Returns:
            Trajectory: The default trajectories.

        """
        env_name = env.__class__.__name__
        if "Mjx" in env_name:
            env_name = env_name.replace("Mjx", "")

        if isinstance(default_dataset_conf.task, str):
            default_dataset_conf.task = [default_dataset_conf.task]

        path_to_conf = loco_mujoco.PATH_TO_VARIABLES

        try:
            with open(path_to_conf, "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                try:
                    path_to_convert_default_datasets = data["LOCOMUJOCO_CONVERTED_DEFAULT_PATH"]
                    os.makedirs(path_to_convert_default_datasets, exist_ok=True)
                except KeyError:
                    path_to_convert_default_datasets = None
        except FileNotFoundError:
            path_to_convert_default_datasets = None

        trajs = []
        for task in default_dataset_conf.task:

            file_path = f"{default_dataset_conf.dataset_type}/{env_name}/{task}.npz"

            traj = None
            cached_file_path = None

            # try to load the default dataset from cache
            if path_to_convert_default_datasets:
                cached_file_path = os.path.join(path_to_convert_default_datasets, file_path)
                if os.path.isfile(cached_file_path):
                    print(f"[LocoMuJoCo's Default Dataset Pipeline] INFO: Found converted dataset at: {cached_file_path}.")
                    traj = Trajectory.load(cached_file_path)

            # if the default dataset is not in the cache, download it from the hub
            if traj is None:

                filename = f"DefaultDatasets/{file_path}"

                try:
                    hub_file_path = hf_hub_download(
                        repo_id="robfiras/loco-mujoco-datasets",
                        filename=filename,
                        repo_type="dataset"
                    )
                    traj = Trajectory.load(hub_file_path)
                except (EntryNotFoundError, HfHubHTTPError):
                    # Target env not on HF → retarget on-the-fly from the source env
                    source_env = getattr(env, "default_dataset_source_env",
                                         _DEFAULT_RETARGET_SOURCE_ENV)
                    print(f"[LocoMuJoCo's Default Dataset Pipeline] INFO: "
                          f"{filename} not on HuggingFace. Retargeting from "
                          f"{source_env} on-the-fly.")
                    traj = ImitationFactory._retarget_default_on_the_fly(
                        env_name, task, default_dataset_conf.dataset_type, source_env)

                # extend the motion to the desired length
                if not traj.data.is_complete:
                    traj = extend_motion(env_name, load_robot_conf_file(env_name).env_params, traj)

                # save  to the cache if the cache path is set
                if cached_file_path:
                    os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
                    traj.save(cached_file_path)

            # filter, extend, and interpolate to match the environment
            traj = TrajectoryHandler.process(traj, env._model, env.dt)

            trajs.append(traj)

        trajs = Trajectory.concatenate(trajs)

        return trajs

    @staticmethod
    def _retarget_default_on_the_fly(env_name_target: str, task: str, dataset_type: str,
                                     source_env: str = _DEFAULT_RETARGET_SOURCE_ENV) -> Trajectory:
        """
        Retarget a default dataset on the fly.

        Called when the pre-retargeted dataset for ``env_name_target`` is not
        available on the HuggingFace hub. Downloads the source-env version
        of the same task and retargets it to the target env using the
        SMPL robot configuration of the target env (shipped with LocoMuJoCo or registered under
        ``LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH``).

        Args:
            env_name_target (str): Name of the environment to retarget to.
            task (str): Name of the task.
            dataset_type (str): Type of the default dataset.
            source_env (str): Name of the environment whose dataset is used as the source.

        Returns:
            Trajectory: The retargeted trajectory.

        """
        source_filename = f"DefaultDatasets/{dataset_type}/{source_env}/{task}.npz"
        source_path = hf_hub_download(
            repo_id="robfiras/loco-mujoco-datasets",
            filename=source_filename,
            repo_type="dataset",
        )
        traj_source = Trajectory.load(source_path)
        return retarget_traj_from_robot_to_robot(source_env, traj_source, env_name_target)

    @staticmethod
    def get_amass_traj(env, amass_dataset_conf: AMASSDatasetConf) -> Trajectory:
        """
        Load the AMASS trajectories of the configuration, retargeted to the environment.

        The datasets of ``dataset_group`` and ``rel_dataset_path`` are loaded with
        :func:`~loco_mujoco.smpl.retargeting.load_retargeted_amass_trajectory`, which retargets them if no cached version
        exists, and processed to match the model and the control frequency of the environment.

        Args:
            env: The environment for which the trajectories are loaded.
            amass_dataset_conf (AMASSDatasetConf): The configuration for the AMASS trajectory.

        Returns:
            Trajectory: The AMASS trajectories.

        Raises:
            ValueError: If the ``dataset_group`` is unknown.

        """

        # Determine dataset paths
        dataset_paths = []
        if amass_dataset_conf.dataset_group is not None:
            if amass_dataset_conf.dataset_group == "AMASS_LOCOMOTION_DATASETS":
                dataset_paths.extend(AMASS_LOCOMOTION_DATASETS)
            else:
                raise ValueError(f"Unknown dataset group: {amass_dataset_conf.dataset_group}")
        if amass_dataset_conf.rel_dataset_path is not None:
            dataset_paths.extend(amass_dataset_conf.rel_dataset_path
                                 if isinstance(amass_dataset_conf.rel_dataset_path, (ListConfig, list))
                                 else [amass_dataset_conf.rel_dataset_path])

        # Load AMASS Trajectory
        traj = load_retargeted_amass_trajectory(env.__class__.__name__, dataset_paths)

        # filter, extend, and interpolate to match the environment
        return TrajectoryHandler.process(traj, env._model, env.dt)

    @staticmethod
    def get_lafan1_traj(env, lafan1_dataset_conf: LAFAN1DatasetConf) -> Trajectory:
        """
        Load the LAFAN1 trajectories of the configuration for the environment.

        The datasets of ``dataset_group`` or, if no group is given, ``dataset_name`` are loaded with
        :func:`~loco_mujoco.datasets.humanoids.LAFAN1.load.load_lafan1_trajectory` and processed to match the model and
        the control frequency of the environment.

        Args:
            env: The environment for which the trajectories are loaded.
            lafan1_dataset_conf (LAFAN1DatasetConf): The configuration for the LAFAN1 trajectory.

        Returns:
            Trajectory: The LAFAN1 trajectories.

        Raises:
            ValueError: If the ``dataset_group`` is unknown.

        """
        # Determine dataset paths
        if lafan1_dataset_conf.dataset_group:
            if lafan1_dataset_conf.dataset_group == "LAFAN1_LOCOMOTION_DATASETS":
                dataset_paths = LAFAN1_LOCOMOTION_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_DANCE_DATASETS":
                dataset_paths = LAFAN1_DANCE_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_ALL_DATASETS":
                dataset_paths = LAFAN1_ALL_DATASETS
            else:
                raise ValueError(f"Unknown dataset group: {lafan1_dataset_conf.dataset_group}")
        else:
            dataset_paths = lafan1_dataset_conf.dataset_name \
                if isinstance(lafan1_dataset_conf.dataset_name, (ListConfig, list)) \
                else [lafan1_dataset_conf.dataset_name]

        # Load LAFAN1 Trajectory
        source_env = getattr(env, "lafan1_dataset_source_env", _LAFAN1_RETARGET_SOURCE_ENV)
        traj = load_lafan1_trajectory(env.__class__.__name__, dataset_paths,
                                      source_env=source_env)

        # filter, extend, and interpolate to match the environment
        return TrajectoryHandler.process(traj, env._model, env.dt)

    @staticmethod
    def get_custom_dataset(env, custom_dataset_conf: CustomDatasetConf) -> Trajectory:
        """
        Process the custom trajectory of the configuration for the environment.

        The trajectory is processed to match the model and the control frequency of the environment.

        Args:
            env: The environment for which the trajectory is loaded.
            custom_dataset_conf (CustomDatasetConf): The configuration for the custom trajectory.

        Returns:
            Trajectory: The processed custom trajectory.

        """
        env_name = env.__class__.__name__
        traj = custom_dataset_conf.traj
        env_params = {}
        # # extend the motion to the desired length
        # if not traj.data.is_complete:
        #
        #     traj = extend_motion(env_name, env_params, traj)

        # filter, extend, and interpolate to match the environment
        return TrajectoryHandler.process(traj, env._model, env.dt)
