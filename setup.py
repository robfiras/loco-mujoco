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
        if subdir != "":
            files = subdir + data_type
            data_files.append(files)
    return data_files


loco_mujoco_xml_package = 'loco_mujoco.environments.data'
loco_mujoco_data_package = 'datasets'


setup(author="Firas Al-Hafez",
      url="https://github.com/robfiras/loco-mujoco",
      version=__version__,
      packages=[package for package in find_packages()
                if package.startswith('loco_mujoco') or package.startswith('datasets')],
      package_data={
          loco_mujoco_data_package: glob_data_files(loco_mujoco_data_package),
          loco_mujoco_xml_package: glob_data_files(loco_mujoco_xml_package)
      }
      )
