"""
Core of LocoMuJoCo: the simulation base classes for MuJoCo and MJX and the modular components they are built from.

The components (observations, rewards, control functions, domain randomizers, terrains, initial and terminal
state handlers, trajectory handling, visualization and environment wrappers) live in the subpackages.

"""

from .utils import MDPInfo, Box, assert_backend_is_supported
from .mujoco_base import Mujoco
from .mujoco_mjx import Mjx, MjxState
from .wrappers import *
from .stateful_object import StatefulObject, EmptyState
from .observations import ObservationContainer, Observation, ObservationType
from .visuals import MujocoViewer, UsdTrajectoryExporter, VideoRecorder
from .trajectory import (Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData,
                         TrajectoryTransitions, TrajectoryHandler, TrajState,
                         interpolate_trajectories)
