from .base import Observation, ObservationIndexContainer, ObservationContainer, ObservationType, StatefulObservation
from .goals import *
from .height_scan import HeightScan

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()
TrajFutureFrames.register()
RootOriRelTraj.register()
TrackingError.register()
HeightScan.register()
