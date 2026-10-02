from typing import Union, List, Tuple
import numpy as np
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class UnitreeGo2(BaseRobotQuadruped):

    """
    Description
    ------------
    Mujoco environment of Unitree Go2 model.


    Default Observation Space
    -----------------

    ============ ================= ================ ==================================== ============================== ===
    Index in Obs Name              ObservationType  Min                                  Max                            Dim
    ============ ================= ================ ==================================== ============================== ===
    0 - 4        q_root            FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_FR_hip_joint    JointPos         [-1.0472]                            [1.0472]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_FR_thigh_joint  JointPos         [-1.5708]                            [3.4907]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_FR_calf_joint   JointPos         [-2.7227]                            [-0.83776]                     1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_FL_hip_joint    JointPos         [-1.0472]                            [1.0472]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_FL_thigh_joint  JointPos         [-1.5708]                            [3.4907]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_FL_calf_joint   JointPos         [-2.7227]                            [-0.83776]                     1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_RR_hip_joint    JointPos         [-1.0472]                            [1.0472]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_RR_thigh_joint  JointPos         [-0.5236]                            [4.5379]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_RR_calf_joint   JointPos         [-2.7227]                            [-0.83776]                     1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_RL_hip_joint    JointPos         [-1.0472]                            [1.0472]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_RL_thigh_joint  JointPos         [-0.5236]                            [4.5379]                       1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_RL_calf_joint   JointPos         [-2.7227]                            [-0.83776]                     1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    17 - 22      dq_root           FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    23           dq_FR_hip_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    24           dq_FR_thigh_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    25           dq_FR_calf_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    26           dq_FL_hip_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    27           dq_FL_thigh_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    28           dq_FL_calf_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    29           dq_RR_hip_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    30           dq_RR_thigh_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    31           dq_RR_calf_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    32           dq_RL_hip_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    33           dq_RL_thigh_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_RL_calf_joint  JointVel         [-inf]                               [inf]                          1
    ============ ================= ================ ==================================== ============================== ===


    Default Action Space
    ------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0               -1.0 1.0
    --------------- ---- ---
    1               -1.0 1.0
    --------------- ---- ---
    2               -1.0 1.0
    --------------- ---- ---
    3               -1.0 1.0
    --------------- ---- ---
    4               -1.0 1.0
    --------------- ---- ---
    5               -1.0 1.0
    --------------- ---- ---
    6               -1.0 1.0
    --------------- ---- ---
    7               -1.0 1.0
    --------------- ---- ---
    8               -1.0 1.0
    --------------- ---- ---
    9               -1.0 1.0
    --------------- ---- ---
    10              -1.0 1.0
    --------------- ---- ---
    11              -1.0 1.0
    =============== ==== ===

    """

    mjx_enabled = False

    def __init__(self, spec=None, camera_params=None,
                 observation_spec=None, actuation_spec=None, **kwargs):
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            camera_params (dict): Dictionary defining some of the camera parameters for visualization.
            observation_spec (List[ObservationType]): Observation specification.
            actuation_spec (List[str]): Action specification.
            **kwargs: Additional arguments
        """

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # set init position
        if "init_state_type" not in kwargs.keys():
            kwargs["init_state_type"] = "DefaultInitialStateHandler"
            kwargs["init_state_params"] = (dict(qpos_init=self.init_qpos, qvel_init=self.init_qvel))

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        if camera_params is None:
            # make the camera by default a bit higher
            camera_params = dict(follow=dict(distance=3.5, elevation=-20.0, azimuth=90.0))

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec,
                         camera_params=camera_params, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[ObservationType]: A list of observations.
        """

        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            # --- Front ---
            ObservationType.JointPos("q_FR_hip_joint", xml_name="FR_hip_joint"),
            ObservationType.JointPos("q_FR_thigh_joint", xml_name="FR_thigh_joint"),
            ObservationType.JointPos("q_FR_calf_joint", xml_name="FR_calf_joint"),
            ObservationType.JointPos("q_FL_hip_joint", xml_name="FL_hip_joint"),
            ObservationType.JointPos("q_FL_thigh_joint", xml_name="FL_thigh_joint"),
            ObservationType.JointPos("q_FL_calf_joint", xml_name="FL_calf_joint"),
            # --- Rear ---
            ObservationType.JointPos("q_RR_hip_joint", xml_name="RR_hip_joint"),
            ObservationType.JointPos("q_RR_thigh_joint", xml_name="RR_thigh_joint"),
            ObservationType.JointPos("q_RR_calf_joint", xml_name="RR_calf_joint"),
            ObservationType.JointPos("q_RL_hip_joint", xml_name="RL_hip_joint"),
            ObservationType.JointPos("q_RL_thigh_joint", xml_name="RL_thigh_joint"),
            ObservationType.JointPos("q_RL_calf_joint", xml_name="RL_calf_joint"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            # --- Front ---
            ObservationType.JointVel("dq_FR_hip_joint", xml_name="FR_hip_joint"),
            ObservationType.JointVel("dq_FR_thigh_joint", xml_name="FR_thigh_joint"),
            ObservationType.JointVel("dq_FR_calf_joint", xml_name="FR_calf_joint"),
            ObservationType.JointVel("dq_FL_hip_joint", xml_name="FL_hip_joint"),
            ObservationType.JointVel("dq_FL_thigh_joint", xml_name="FL_thigh_joint"),
            ObservationType.JointVel("dq_FL_calf_joint", xml_name="FL_calf_joint"),
            # --- Rear ---
            ObservationType.JointVel("dq_RR_hip_joint", xml_name="RR_hip_joint"),
            ObservationType.JointVel("dq_RR_thigh_joint", xml_name="RR_thigh_joint"),
            ObservationType.JointVel("dq_RR_calf_joint", xml_name="RR_calf_joint"),
            ObservationType.JointVel("dq_RL_hip_joint", xml_name="RL_hip_joint"),
            ObservationType.JointVel("dq_RL_thigh_joint", xml_name="RL_thigh_joint"),
            ObservationType.JointVel("dq_RL_calf_joint", xml_name="RL_calf_joint")]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """
        action_spec = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        Returns:
            str: The default path to the xml file.
        """
        return cls.get_model_path("unitree_go2", "go2.xml")

    @info_property
    def grf_size(self) -> int:
        """
        Returns the size of the ground force vector.

        Returns:
            int: The size of the ground force vector.
        """
        return 12

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml.

        Returns:
            str: The name of the upper body.
        """
        return self.root_body_name

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml.

        Returns:
            str: The name of the root body.
        """
        return "base"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root body in the Mujoco xml.

        Returns:
            str: The name of the free joint.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: The healthy range of the root height.
        """
        return (0.25, 1.0)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geometries.

        Returns:
            List[str]: The names of the foot geometries.
        """
        return ["RL_foot", "RR_foot", "FL_foot", "FR_foot"]

    @info_property
    def init_qpos(self) -> np.ndarray:
        """
        Returns the initial joint positions.

        Returns:
            np.ndarray: The initial joint positions.
        """
        return np.array([0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.8, 0.0,
                         0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8])

    @info_property
    def init_qvel(self) -> np.ndarray:
        """
        Returns the initial joint velocities.

        Returns:
            np.ndarray: The initial joint velocities.
        """
        return np.zeros(18)