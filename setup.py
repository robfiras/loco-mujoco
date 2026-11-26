from setuptools import setup, find_packages
import glob
from loco_mujoco import __version__


def glob_data_files(data_package, data_type=None):
    data_type = '*' if data_type is None else data_type
    data_dir = data_package.replace(".", "/")
    data_files = []
    directories = glob.glob(data_dir+'/**/', recursive=True)
    for directory in directories:
        subdir = directory[len(data_dir)+1:]
        # bytecode caches of a working copy are not data; without this check they end up in the wheel
        if subdir != "" and "__pycache__" not in subdir:
            files = subdir + data_type
            data_files.append(files)
    return data_files


loco_mujoco_xml_package = 'loco_mujoco.models'
loco_mujoco_smpl_package = 'loco_mujoco.smpl'


setup(author="Firas Al-Hafez",
      url="https://github.com/robfiras/loco-mujoco",
      version=__version__,
      packages=[package for package in find_packages()
                if package.startswith('loco_mujoco') or package.startswith('datasets')],
      package_data={
          loco_mujoco_xml_package: glob_data_files(loco_mujoco_xml_package),
          loco_mujoco_smpl_package: glob_data_files(loco_mujoco_smpl_package)
      }
      )
