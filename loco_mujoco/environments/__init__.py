"""
Environments of LocoMuJoCo: the base class :class:`~loco_mujoco.environments.base.LocoEnv` and the humanoid and
quadruped environments built on it. Importing this package registers all environments.
"""

from .base import LocoEnv
from .humanoids import *
from .quadrupeds import *
