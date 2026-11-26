"""Observations: the observation types, the container that assembles them and the goal observations."""

from .base import (Observation, ObservationIndexContainer, ObservationContainer, ObservationType,
                   StatefulObservation, HeightScan)
from .goals import *

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()
TrajFutureFrames.register()
RootOriRelTraj.register()
TrackingError.register()
