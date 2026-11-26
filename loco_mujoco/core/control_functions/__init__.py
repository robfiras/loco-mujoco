"""Control functions, which map the actions of a policy to the controls of the simulation."""

from .base import ControlFunction
from .default import DefaultControl
from .pd import PDControl

# register all control functions
DefaultControl.register()
PDControl.register()
