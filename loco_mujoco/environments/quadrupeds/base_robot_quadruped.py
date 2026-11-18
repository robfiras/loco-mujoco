"""
Base class of the quadruped environments.
"""

from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils import info_property


class BaseRobotQuadruped(LocoEnv):
    """
    Base class of the quadruped robot environments.

    It extends :class:`~loco_mujoco.environments.base.LocoEnv` with the defaults shared by the quadrupeds. The
    robot-specific subclasses define the model, the default observation and action spaces and further info
    properties.

    """

    @info_property
    def sites_for_mimic(self):
        """
        Returns the default sites that are used for mimic. No such sites are defined for the quadrupeds.

        Returns:
            List[str]: An empty list.

        """
        return []

    @info_property
    def root_body_name(self):
        """
        Returns the name of the root body of the robot in the XML file.

        Returns:
            str: Name of the root body.

        """
        return "trunk"
