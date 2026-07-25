import os
from pathlib import Path
from dataclasses import replace
from typing import Union, List

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, OfflineModeIsEnabled
import httpx
import requests

import loco_mujoco
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.datasets.data_generation import ExtendTrajData, optimize_for_collisions, calculate_qvel_with_finite_difference
from loco_mujoco.smpl.retargeting import load_robot_conf_file, retarget_traj_from_robot_to_robot
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories)
from loco_mujoco.utils import setup_logger


# Source env used when retargeting on-the-fly for an env that is not present
# on the HuggingFace dataset repo.
_LAFAN1_RETARGET_SOURCE_ENV = "UnitreeH1v2"


def extend_motion(
        env_name: str,
        robot_conf: DictConfig,
        traj: Trajectory,
        replace_qvel_with_finite_diff: bool,
) -> Trajectory:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        traj (Trajectory): The original trajectory data.

    Returns:
        Trajectory: The extended trajectory.

    """

    assert traj.data.n_trajectories == 1

    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    traj_data, traj_info = interpolate_trajectories(traj.data, traj.info, 1.0 / env.dt)
    traj = Trajectory(info=traj_info, data=traj_data)

    env.process_trajectory(traj)
    traj_data, traj_info = env._traj.data, env._traj.info

    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(
        n_episodes=env.th.n_trajectories(env._traj.data),
        render=False,
        callback_class=callback
    )
    # extend_trajectory_data asserts current_length == qpos.shape[0]; process_trajectory
    # pads qpos to max_n_samples, so trim before passing it in.
    traj_data = traj_data.trim()
    traj_data, traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    traj = replace(traj, data=traj_data, info=traj_info)

    return traj


def load_lafan1_trajectory(
        env_name: str,
        dataset_name: Union[str, List[str]],
        max_steps: int = 100,
        source_env: str = _LAFAN1_RETARGET_SOURCE_ENV,
) -> Trajectory:
    """
    Load a trajectory from the LAFAN1 dataset.

    Args:
        env_name (str): The name of the environment.
        dataset_name (Union[str, List[str]]): The name of the dataset(s) to load.
        max_steps (int, optional): The maximum number of steps to optimize for collisions. Defaults to 100.
        source_env (str, optional): Source env used when `env_name`'s LAFAN1 dataset
            is not on HuggingFace and retargeting on-the-fly is needed.

    Returns:
        Trajectory: The loaded trajectory.

    """
    logger = setup_logger("lafan1", identifier="[LocoMuJoCo's LAFAN1 Retargeting Pipeline]")

    if "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    path_to_conf = loco_mujoco.PATH_TO_VARIABLES

    try:
        with open(path_to_conf, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            try:
                path_to_convert_lafan1_datasets = data["LOCOMUJOCO_CONVERTED_LAFAN1_PATH"]
            except KeyError:
                path_to_convert_lafan1_datasets = None
    except FileNotFoundError:
        path_to_convert_lafan1_datasets = None

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for d_name in dataset_name:

        # Strip the .npz suffix once up-front so it's consistent throughout
        # the loop (and for the recursive retargeting call).
        if d_name.endswith(".npz"):
            d_name = d_name[:-len(".npz")]

        if path_to_convert_lafan1_datasets:
            target_path_dataset = os.path.join(path_to_convert_lafan1_datasets, env_name, f"{d_name}.npz")
        else:
            target_path_dataset = None

        if path_to_convert_lafan1_datasets:
            # check if file exists
            if os.path.exists(target_path_dataset):
                logger.info(f"Found converted dataset at: {target_path_dataset}.")
                traj = Trajectory.load(target_path_dataset)
                all_trajectories.append(traj)
                continue

        # load the npz file
        hf_filename = f"{d_name}.npz"

        try:
            file_path = hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=f"Lafan1/mocap/{env_name}/{hf_filename}",
                repo_type="dataset"
            )
            traj = Trajectory.load(file_path)
        except (EntryNotFoundError, HfHubHTTPError,
                requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                httpx.TransportError, httpx.HTTPError,
                OfflineModeIsEnabled):
            # Target env not on HF (or HF unreachable) → retarget from source env
            logger.info(f"Lafan1/mocap/{env_name}/{hf_filename} not on HuggingFace. "
                        f"Retargeting from {source_env} on-the-fly.")
            traj_source = load_lafan1_trajectory(source_env, [d_name],
                                                 max_steps=max_steps,
                                                 source_env=source_env)
            traj = retarget_traj_from_robot_to_robot(source_env, traj_source, env_name)

        # extend the motion to the desired length
        if not traj.data.is_complete:
            logger.info("Using Mujoco's kinematics to calculate other model-specific entities ...")
            traj = extend_motion(env_name, load_robot_conf_file(env_name), traj, replace_qvel_with_finite_diff=False)

        if path_to_convert_lafan1_datasets:
            os.makedirs(os.path.dirname(target_path_dataset), exist_ok=True)
            traj.save(target_path_dataset)

        all_trajectories.append(traj)

    # concatenate trajectories
    if len(all_trajectories) == 1:
        trajectory = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=np)
        trajectory = Trajectory(traj_info, traj_data)

    logger.info("Trajectory data loaded!")

    return trajectory


if __name__ == "__main__":
    from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_ALL_DATASETS

    traj = load_lafan1_trajectory("UnitreeH1", LAFAN1_ALL_DATASETS)
    print("Done!")
