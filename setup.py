from setuptools import setup, find_packages
from os import path

requires_list = []
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(name='loco_mujoco',
      version='0.1',
      description='Imitation learning benchmark focusing on complex locomotion tasks using MuJoCo.',
      license='MIT',
      author="Firas Al-Hafez",
      author_mail="fi.alhafez@gmail.com",
      packages=[package for package in find_packages()
                if package.startswith('loco_mujoco')],
      install_requires=requires_list,
      zip_safe=False,
      )
