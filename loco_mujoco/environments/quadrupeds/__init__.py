"""
Quadruped robot environments.

Importing this package registers the Unitree A1, Unitree Go2, Boston Dynamics Spot and ANYmal C environments, each
in a CPU (MuJoCo) and an Mjx version, so that they can be created by name through the task factories.
"""

from .base_robot_quadruped import BaseRobotQuadruped
from .unitreeA1 import UnitreeA1
from .unitreeA1_mjx import MjxUnitreeA1
from .unitreeGo2 import UnitreeGo2
from .unitreeGo2_mjx import MjxUnitreeGo2
from .bd_spot import BDSpot
from .bd_spot_mjx import MjxBDSpot
from .anymal_c import AnymalC
from .anymal_c_mjx import MjxAnymalC


# register environment
UnitreeA1.register()
MjxUnitreeA1.register()
UnitreeGo2.register()
MjxUnitreeGo2.register()
BDSpot.register()
MjxBDSpot.register()
AnymalC.register()
MjxAnymalC.register()


