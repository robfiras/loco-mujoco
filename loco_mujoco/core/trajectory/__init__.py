"""
Trajectory data structures and handlers.

Contains the dataclasses that store reference motions (:class:`~loco_mujoco.core.trajectory.dataclasses.Trajectory`
and its parts) and the :class:`~loco_mujoco.core.trajectory.handler.TrajectoryHandler` classes, which track the
position of an environment in its trajectory and decide where an episode starts.

"""

from .dataclasses import (Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData, TrajectoryTransitions,
                          interpolate_trajectories)
from .handler import (TrajectoryHandler, TrajState,
                      RandomStartTrajectoryHandler,
                      RandomTrajFixedStepTrajectoryHandler,
                      FixedStartTrajectoryHandler)

RandomStartTrajectoryHandler.register()
RandomTrajFixedStepTrajectoryHandler.register()
FixedStartTrajectoryHandler.register()
