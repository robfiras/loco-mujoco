from .base import Observation, ObservationIndexContainer, ObservationContainer, ObservationType, StatefulObservation
from .goals import *
from .goals_future import TrajFutureFrames, RootOriRelTraj, TrackingError

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()
TrajFutureFrames.register()
RootOriRelTraj.register()
TrackingError.register()
