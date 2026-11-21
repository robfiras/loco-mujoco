"""
Base class of the humanoid robot environments.
"""

from typing import List, Tuple
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils import info_property


class BaseRobotHumanoid(LocoEnv):
    """
    Base class for the humanoid robot environments.

    It extends :class:`~loco_mujoco.environments.base.LocoEnv` with the defaults shared by all humanoid robots:
    the names of the mimic sites, the name of the root body, the offset of the goal visualization arrow, and
    the environments from which default and LAFAN1 datasets are retargeted when no dataset is available for a
    robot. The robot-specific model, observation space and action space are defined in the subclasses.

    """

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Returns the default sites that are used for mimic.

        These sites are attached to corresponding body parts on every humanoid model and are used, for
        example, by the site-based mimic rewards and observations and for retargeting.

        Returns:
            List[str]: List of site names.

        """
        return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
                "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
                "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
                "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body of the robot in the MuJoCo xml.

        Returns:
            str: Name of the root body.

        """
        return "pelvis"

    @info_property
    def goal_visualization_arrow_offset(self) -> Tuple[float, float, float]:
        """
        Returns the offset for the goal visualization arrow. This corresponds to the offset from the
        root body of the robot.

        Returns:
            Tuple[float, float, float]: Offset ``[x, y, z]`` in meters.

        """
        return [0, 0, 0.6]

    @info_property
    def default_dataset_source_env(self) -> str:
        """
        Returns the source environment used when retargeting a default dataset for this environment on the fly.

        This is used when ``DefaultDatasets/.../<this_env>/<task>.npz`` is not available on the hub.
        Subclasses can override this to choose a retargeting source that matches their skeleton more closely.

        Returns:
            str: Name of the source environment.

        """
        return "SkeletonTorque"

    @info_property
    def lafan1_dataset_source_env(self) -> str:
        """
        Returns the source environment used when retargeting a LAFAN1 dataset for this environment on the fly.

        Defaults to ``UnitreeH1v2`` (the environment whose LAFAN1 motions were hand-converted and published on
        the loco-mujoco HuggingFace repository). Subclasses can override this if a different source environment
        gives better retargeting quality.

        Returns:
            str: Name of the source environment.

        """
        return "UnitreeH1v2"
