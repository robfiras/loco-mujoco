import os
import wget
import zipfile
import numpy as np
from copy import deepcopy
from pathlib import Path
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import loco_mujoco


def download_all_datasets():
    """
    Download and installs all datasets.

    """
    download_real_datasets()
    download_perfect_datasets()


def download_real_datasets():
    """
    Download and installs real datasets.

    """

    dataset_path = Path(loco_mujoco.__file__).resolve().parent / "datasets"
    print(dataset_path)

    print("\nDownloading Humanoid Datasets ...")
    dataset_path_humanoid = dataset_path / "humanoids/real"
    dataset_path_humanoid_str = str(dataset_path_humanoid)
    os.makedirs(dataset_path_humanoid_str, exist_ok=True)
    humanoid_url = "https://zenodo.org/records/11217638/files/humanoid_datasets_v0.3.0.zip?download=1"
    wget.download(humanoid_url, out=dataset_path_humanoid_str)
    print("\n")
    file_name = "humanoid_datasets_v0.3.0.zip"
    file_path = str(dataset_path_humanoid / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_humanoid_str)
    os.remove(file_path)

    print("\nDownloading Quadruped Datasets ...")
    dataset_path_quadrupeds = dataset_path / "quadrupeds/real"
    dataset_path_quadrupeds_str = str(dataset_path_quadrupeds)
    os.makedirs(dataset_path_quadrupeds_str, exist_ok=True)
    quadruped_url = "https://zenodo.org/records/11217638/files/quadruped_datasets_v0.3.0.zip?download=1"
    wget.download(quadruped_url, out=dataset_path_quadrupeds_str)
    print("\n")
    file_name = "quadruped_datasets_v0.3.0.zip"
    file_path = str(dataset_path_quadrupeds / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_quadrupeds_str)
    os.remove(file_path)


def download_perfect_datasets():
    """
    Download and installs perfect datasets.

    """
    dataset_path = Path(loco_mujoco.__file__).resolve().parent / "datasets"

    print("\nDownloading Perfect Humanoid Datasets ...")
    dataset_path_humanoid = dataset_path / "humanoids/perfect"
    dataset_path_humanoid_str = str(dataset_path_humanoid)
    os.makedirs(dataset_path_humanoid_str, exist_ok=True)
    humanoid_url = "https://zenodo.org/records/11217638/files/humanoid_datasets_perfect_v0.3.0.zip?download=1"
    wget.download(humanoid_url, out=dataset_path_humanoid_str)
    print("\n")
    file_name = "humanoid_datasets_perfect_v0.3.0.zip"
    file_path = str(dataset_path_humanoid / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_humanoid_str)
    os.remove(file_path)


    print("\nDownloading Perfect Quadruped Datasets ...")
    dataset_path_quadrupeds = dataset_path / "quadrupeds/perfect"
    dataset_path_quadrupeds_str = str(dataset_path_quadrupeds)
    os.makedirs(dataset_path_quadrupeds_str, exist_ok=True)
    quadruped_url = "https://zenodo.org/records/11217638/files/quadruped_datasets_perfect_v0.3.0.zip?download=1"
    wget.download(quadruped_url, out=dataset_path_quadrupeds_str)
    print("\n")
    file_name = "quadruped_datasets_perfect_v0.3.0.zip"
    file_path = str(dataset_path_quadrupeds / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_quadrupeds_str)
    os.remove(file_path)


def download_raw_mocap_datasets():
    """
    Download and installs raw mocap datasets, which are not optimized for any specific humanoid model.

    """

    dataset_path = Path(loco_mujoco.__file__).resolve().parent / "datasets"
    print(dataset_path)

    print("\nDownloading Raw Mocap Datasets ...")
    dataset_path_humanoid = dataset_path / "data_generation/00_raw_mocap_data"
    dataset_path_humanoid_str = str(dataset_path_humanoid)
    os.makedirs(dataset_path_humanoid_str, exist_ok=True)
    humanoid_url = "https://zenodo.org/records/10625721/files/raw_motion_capture_v0.1.zip?download=1"
    wget.download(humanoid_url, out=dataset_path_humanoid_str)
    print("\n")
    file_name = "raw_motion_capture_v0.1.zip"
    file_path = str(dataset_path_humanoid / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_humanoid_str)
    os.remove(file_path)


def adapt_mocap(path, joint_conf, unavailable_keys, rename_map=None, discard_first=None, discard_last=None):
    """
    Applies a linear transformation to the joint angles and velocities of a mocap dataset to adapt to specific
    humanoid model.

    Args:
        path (str): Path to the mocap dataset.
        joint_conf (dict): Dictionary containing the multipliers and offsets for linear transformation of the joint.
        unavailable_keys (list or dict): List of joint names that are not available in the mocap dataset. If a
         list is provided, the joint angles and velocities are set to zero. If a dictionary is provided, the joint
         positions are set to the values in the dictionary and the velocities are set to zero.
        rename_map (dict): Dictionary containing the mapping of joint names in the mocap dataset to the
         joint names provided.
        discard_first (int): Number of initial obs to discard.
        discard_last (int): Number of final obs to discard.

    Returns:
         Dictionary containing the joint angles and velocities of the modified mocap dataset.

    """

    # extract the euler keys
    euler_keys = list(joint_conf.keys())

    # extract the multipliers and offsets for linear transformation
    multipliers = [joint_conf[k][0] for k in euler_keys]
    offsets = [joint_conf[k][1] for k in euler_keys]

    # load the mocap dataset
    data = sio.loadmat(path)
    joint_pos = data["angJoi"]
    joint_vel = data["angDJoi"]
    try:
        joint_names = data["rowNameIK"]
    except:
        joint_names = data["rowName"]
    joint_names = np.array([name[0] for name in np.squeeze(joint_names)])

    n_datapoint = len(joint_pos[0])
    joint_pos = dict(zip(joint_names, joint_pos))
    joint_vel = dict(zip(joint_names, joint_vel))
    if type(unavailable_keys) == list:
        for ukey in unavailable_keys:
            joint_pos[ukey] = np.zeros(n_datapoint)
            joint_vel[ukey] = np.zeros(n_datapoint)
    elif type(unavailable_keys) == dict:
        for ukey, val in unavailable_keys.items():
            joint_pos[ukey] = np.ones(n_datapoint) * val
            joint_vel[ukey] = np.zeros(n_datapoint)
    else:
        raise TypeError

    # get the relevant data
    joint_pos = np.array([joint_pos[k] for k in euler_keys])
    joint_vel = np.array([joint_vel[k] for k in euler_keys])

    # apply multipliers and offsets
    multipliers = np.transpose(np.tile(np.array(multipliers), (np.shape(joint_pos)[1], 1)))
    offsets = np.transpose(np.tile(offsets, (np.shape(joint_pos)[1], 1)))
    joint_pos = joint_pos * multipliers + offsets
    joint_vel = joint_vel * multipliers

    # combine joint positions and velocities
    trajec = np.concatenate((joint_pos, joint_vel))

    # rename if needed
    if rename_map is not None:
        for k, v in rename_map.items():
            i = euler_keys.index(k)
            euler_keys[i] = v

    keys = ["q_" + k for k in euler_keys] + ["dq_" + k for k in euler_keys]

    # add goal if available
    if "goal" in data.keys():
        keys.append("goal")
        goal = data["goal"]
        trajec = np.concatenate((trajec, goal))

    # create dataset
    dataset = dict(zip(keys, trajec))

    # if needed discard first and last part of the dataset
    for j_name, val in dataset.items():
        val_temp = val[discard_first:]
        val_temp = val_temp[0:-discard_last]
        dataset[j_name] = val_temp

    return dataset
