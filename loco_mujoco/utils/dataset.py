import os
import wget
import zipfile
from pathlib import Path
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

    print("\nDownloading Humanoid Datasets ...\n")
    dataset_path_humanoid = dataset_path / "humanoids/real"
    dataset_path_humanoid_str = str(dataset_path_humanoid)
    os.makedirs(dataset_path_humanoid_str, exist_ok=True)
    humanoid_url = "https://zenodo.org/records/10102870/files/humanoid_datasets_v0.1.zip?download=1"
    wget.download(humanoid_url, out=dataset_path_humanoid_str)
    file_name = "humanoid_datasets_v0.1.zip"
    file_path = str(dataset_path_humanoid / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_humanoid_str)
    os.remove(file_path)

    print("\nDownloading Quadruped Datasets ...\n")
    dataset_path_quadrupeds = dataset_path / "quadrupeds/real"
    dataset_path_quadrupeds_str = str(dataset_path_quadrupeds)
    os.makedirs(dataset_path_quadrupeds_str, exist_ok=True)
    quadruped_url = "https://zenodo.org/records/10102870/files/quadruped_datasets_v0.1.zip?download=1"
    wget.download(quadruped_url, out=dataset_path_quadrupeds_str)
    file_name = "quadruped_datasets_v0.1.zip"
    file_path = str(dataset_path_quadrupeds / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_quadrupeds_str)
    os.remove(file_path)


def download_perfect_datasets():
    """
    Download and installs perfect datasets.

    """
    dataset_path = Path(loco_mujoco.__file__).resolve().parent / "datasets"

    print("\nDownloading Perfect Humanoid Datasets ...\n")
    dataset_path_humanoid = dataset_path / "humanoids/perfect"
    dataset_path_humanoid_str = str(dataset_path_humanoid)
    os.makedirs(dataset_path_humanoid_str, exist_ok=True)
    humanoid_url = "https://zenodo.org/records/10393490/files/humanoid_datasets_perfect_v0.1.zip?download=1"
    wget.download(humanoid_url, out=dataset_path_humanoid_str)
    file_name = "humanoid_datasets_perfect_v0.1.zip"
    file_path = str(dataset_path_humanoid / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_humanoid_str)
    os.remove(file_path)


    print("\nDownloading Perfect Quadruped Datasets ...\n")
    dataset_path_quadrupeds = dataset_path / "quadrupeds/perfect"
    dataset_path_quadrupeds_str = str(dataset_path_quadrupeds)
    os.makedirs(dataset_path_quadrupeds_str, exist_ok=True)
    quadruped_url = "https://zenodo.org/records/10393490/files/quadruped_datasets_perfect_v0.1.zip?download=1"
    wget.download(quadruped_url, out=dataset_path_quadrupeds_str)
    file_name = "quadruped_datasets_perfect_v0.1.zip"
    file_path = str(dataset_path_quadrupeds / file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path_quadrupeds_str)
    os.remove(file_path)

