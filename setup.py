from setuptools import setup, find_packages

requires_list = ["mushroom_rl>=1.9.0"]

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
