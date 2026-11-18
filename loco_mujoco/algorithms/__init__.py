"""
JAX/Flax learning algorithms of LocoMuJoCo.

The core algorithms are :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax`,
:class:`~loco_mujoco.algorithms.gail_jax.GAILJax`, :class:`~loco_mujoco.algorithms.amp_jax.AMPJax` and
:class:`~loco_mujoco.algorithms.supermimic_jax.SuperMimicJax`; DeepMimic-style tracking is PPO combined with
:class:`~loco_mujoco.core.reward.trajectory_based.MimicReward`. Further algorithms are in the
:mod:`~loco_mujoco.algorithms.experimental` subpackage. The shared base classes, networks and data containers
of the ``common`` subpackage are re-exported here. Training examples are in ``examples/training_examples``.

"""

from .common import *
from loco_mujoco.algorithms.common.networks import FullyConnectedNet, ActorCritic, RunningMeanStd
from .ppo_jax import PPOJax
from .gail_jax import GAILJax
from .amp_jax import AMPJax
from . import experimental
from .supermimic_jax import SuperMimicJax
