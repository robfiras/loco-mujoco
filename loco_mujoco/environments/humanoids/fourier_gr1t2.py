from typing import Union, List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class FourierGR1T2(BaseRobotHumanoid):

    """

    Description
    ------------



    Default Observation Space
    -------------------------
    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 4        q_root                        FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_joint_left_hip_roll         JointPos         [-0.09]                              [0.79]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_joint_left_hip_yaw          JointPos         [-0.7]                               [0.7]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_joint_left_hip_pitch        JointPos         [-1.75]                              [0.7]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_joint_left_knee_pitch       JointPos         [-0.09]                              [1.92]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_joint_left_ankle_pitch      JointPos         [-1.05]                              [0.52]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_joint_left_ankle_roll       JointPos         [-0.44]                              [0.44]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_joint_right_hip_roll        JointPos         [-0.79]                              [0.09]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_joint_right_hip_yaw         JointPos         [-0.7]                               [0.7]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_joint_right_hip_pitch       JointPos         [-1.75]                              [0.7]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_joint_right_knee_pitch      JointPos         [-0.09]                              [1.92]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_joint_right_ankle_pitch     JointPos         [-1.05]                              [0.52]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_joint_right_ankle_roll      JointPos         [-0.44]                              [0.44]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_joint_waist_yaw             JointPos         [-1.05]                              [1.05]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_joint_waist_pitch           JointPos         [-0.52]                              [1.22]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_joint_waist_roll            JointPos         [-0.7]                               [0.7]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_joint_head_pitch            JointPos         [-0.87]                              [0.87]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_joint_head_roll             JointPos         [-0.35]                              [0.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_joint_head_yaw              JointPos         [-2.71]                              [2.71]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_joint_left_shoulder_pitch   JointPos         [-2.79]                              [1.92]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_joint_left_shoulder_roll    JointPos         [-0.57]                              [3.27]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_joint_left_shoulder_yaw     JointPos         [-2.97]                              [2.97]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_joint_left_elbow_pitch      JointPos         [-2.27]                              [2.27]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_joint_left_wrist_yaw        JointPos         [-2.97]                              [2.97]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_joint_left_wrist_roll       JointPos         [-0.96]                              [0.87]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_joint_left_wrist_pitch      JointPos         [-0.61]                              [0.61]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_joint_right_shoulder_pitch  JointPos         [-2.79]                              [1.92]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_joint_right_shoulder_roll   JointPos         [-3.27]                              [0.57]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_joint_right_shoulder_yaw    JointPos         [-2.97]                              [2.97]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_joint_right_elbow_pitch     JointPos         [-2.27]                              [2.27]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_joint_right_wrist_yaw       JointPos         [-2.97]                              [2.97]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           q_joint_right_wrist_roll      JointPos         [-0.87]                              [0.96]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           q_joint_right_wrist_pitch     JointPos         [-0.61]                              [0.61]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37 - 42      dq_root                       FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_joint_left_hip_roll        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_joint_left_hip_yaw         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_joint_left_hip_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_joint_left_knee_pitch      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_joint_left_ankle_pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_joint_left_ankle_roll      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_joint_right_hip_roll       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_joint_right_hip_yaw        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_joint_right_hip_pitch      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_joint_right_knee_pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_joint_right_ankle_pitch    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_joint_right_ankle_roll     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_joint_waist_yaw            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_joint_waist_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_joint_waist_roll           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_joint_head_pitch           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_joint_head_roll            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_joint_head_yaw             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    61           dq_joint_left_shoulder_pitch  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    62           dq_joint_left_shoulder_roll   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    63           dq_joint_left_shoulder_yaw    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    64           dq_joint_left_elbow_pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    65           dq_joint_left_wrist_yaw       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    66           dq_joint_left_wrist_roll      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    67           dq_joint_left_wrist_pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    68           dq_joint_right_shoulder_pitch JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    69           dq_joint_right_shoulder_roll  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    70           dq_joint_right_shoulder_yaw   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    71           dq_joint_right_elbow_pitch    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    72           dq_joint_right_wrist_yaw      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    73           dq_joint_right_wrist_roll     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    74           dq_joint_right_wrist_pitch    JointVel         [-inf]                               [inf]                          1
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    --------------------

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
    --------------- ---- ---
    12              -1.0 1.0
    --------------- ---- ---
    13              -1.0 1.0
    --------------- ---- ---
    14              -1.0 1.0
    --------------- ---- ---
    15              -1.0 1.0
    --------------- ---- ---
    16              -1.0 1.0
    --------------- ---- ---
    17              -1.0 1.0
    --------------- ---- ---
    18              -1.0 1.0
    --------------- ---- ---
    19              -1.0 1.0
    --------------- ---- ---
    20              -1.0 1.0
    --------------- ---- ---
    21              -1.0 1.0
    --------------- ---- ---
    22              -1.0 1.0
    --------------- ---- ---
    23              -1.0 1.0
    --------------- ---- ---
    24              -1.0 1.0
    --------------- ---- ---
    25              -1.0 1.0
    --------------- ---- ---
    26              -1.0 1.0
    --------------- ---- ---
    27              -1.0 1.0
    --------------- ---- ---
    28              -1.0 1.0
    --------------- ---- ---
    29              -1.0 1.0
    --------------- ---- ---
    30              -1.0 1.0
    --------------- ---- ---
    31              -1.0 1.0
    =============== ==== ===


    Methods
    ---------

    """

    mjx_enabled = False

    def __init__(self, spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None, **kwargs) -> None:
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the XML file or an `MjSpec` object. If none is provided, the default XML file is used.
            observation_spec (List[Observation]): Observation specification.
            actuation_spec (List[str]): Action specification.
            **kwargs: Additional arguments.
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

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: A list of observation types.

        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_joint_left_hip_roll", xml_name="joint_left_hip_roll"),
            ObservationType.JointPos("q_joint_left_hip_yaw", xml_name="joint_left_hip_yaw"),
            ObservationType.JointPos("q_joint_left_hip_pitch", xml_name="joint_left_hip_pitch"),
            ObservationType.JointPos("q_joint_left_knee_pitch", xml_name="joint_left_knee_pitch"),
            ObservationType.JointPos("q_joint_left_ankle_pitch", xml_name="joint_left_ankle_pitch"),
            ObservationType.JointPos("q_joint_left_ankle_roll", xml_name="joint_left_ankle_roll"),
            ObservationType.JointPos("q_joint_right_hip_roll", xml_name="joint_right_hip_roll"),
            ObservationType.JointPos("q_joint_right_hip_yaw", xml_name="joint_right_hip_yaw"),
            ObservationType.JointPos("q_joint_right_hip_pitch", xml_name="joint_right_hip_pitch"),
            ObservationType.JointPos("q_joint_right_knee_pitch", xml_name="joint_right_knee_pitch"),
            ObservationType.JointPos("q_joint_right_ankle_pitch", xml_name="joint_right_ankle_pitch"),
            ObservationType.JointPos("q_joint_right_ankle_roll", xml_name="joint_right_ankle_roll"),
            ObservationType.JointPos("q_joint_waist_yaw", xml_name="joint_waist_yaw"),
            ObservationType.JointPos("q_joint_waist_pitch", xml_name="joint_waist_pitch"),
            ObservationType.JointPos("q_joint_waist_roll", xml_name="joint_waist_roll"),
            ObservationType.JointPos("q_joint_head_pitch", xml_name="joint_head_pitch"),
            ObservationType.JointPos("q_joint_head_roll", xml_name="joint_head_roll"),
            ObservationType.JointPos("q_joint_head_yaw", xml_name="joint_head_yaw"),
            ObservationType.JointPos("q_joint_left_shoulder_pitch", xml_name="joint_left_shoulder_pitch"),
            ObservationType.JointPos("q_joint_left_shoulder_roll", xml_name="joint_left_shoulder_roll"),
            ObservationType.JointPos("q_joint_left_shoulder_yaw", xml_name="joint_left_shoulder_yaw"),
            ObservationType.JointPos("q_joint_left_elbow_pitch", xml_name="joint_left_elbow_pitch"),
            ObservationType.JointPos("q_joint_left_wrist_yaw", xml_name="joint_left_wrist_yaw"),
            ObservationType.JointPos("q_joint_left_wrist_roll", xml_name="joint_left_wrist_roll"),
            ObservationType.JointPos("q_joint_left_wrist_pitch", xml_name="joint_left_wrist_pitch"),
            ObservationType.JointPos("q_joint_right_shoulder_pitch", xml_name="joint_right_shoulder_pitch"),
            ObservationType.JointPos("q_joint_right_shoulder_roll", xml_name="joint_right_shoulder_roll"),
            ObservationType.JointPos("q_joint_right_shoulder_yaw", xml_name="joint_right_shoulder_yaw"),
            ObservationType.JointPos("q_joint_right_elbow_pitch", xml_name="joint_right_elbow_pitch"),
            ObservationType.JointPos("q_joint_right_wrist_yaw", xml_name="joint_right_wrist_yaw"),
            ObservationType.JointPos("q_joint_right_wrist_roll", xml_name="joint_right_wrist_roll"),
            ObservationType.JointPos("q_joint_right_wrist_pitch", xml_name="joint_right_wrist_pitch"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_joint_left_hip_roll", xml_name="joint_left_hip_roll"),
            ObservationType.JointVel("dq_joint_left_hip_yaw", xml_name="joint_left_hip_yaw"),
            ObservationType.JointVel("dq_joint_left_hip_pitch", xml_name="joint_left_hip_pitch"),
            ObservationType.JointVel("dq_joint_left_knee_pitch", xml_name="joint_left_knee_pitch"),
            ObservationType.JointVel("dq_joint_left_ankle_pitch", xml_name="joint_left_ankle_pitch"),
            ObservationType.JointVel("dq_joint_left_ankle_roll", xml_name="joint_left_ankle_roll"),
            ObservationType.JointVel("dq_joint_right_hip_roll", xml_name="joint_right_hip_roll"),
            ObservationType.JointVel("dq_joint_right_hip_yaw", xml_name="joint_right_hip_yaw"),
            ObservationType.JointVel("dq_joint_right_hip_pitch", xml_name="joint_right_hip_pitch"),
            ObservationType.JointVel("dq_joint_right_knee_pitch", xml_name="joint_right_knee_pitch"),
            ObservationType.JointVel("dq_joint_right_ankle_pitch", xml_name="joint_right_ankle_pitch"),
            ObservationType.JointVel("dq_joint_right_ankle_roll", xml_name="joint_right_ankle_roll"),
            ObservationType.JointVel("dq_joint_waist_yaw", xml_name="joint_waist_yaw"),
            ObservationType.JointVel("dq_joint_waist_pitch", xml_name="joint_waist_pitch"),
            ObservationType.JointVel("dq_joint_waist_roll", xml_name="joint_waist_roll"),
            ObservationType.JointVel("dq_joint_head_pitch", xml_name="joint_head_pitch"),
            ObservationType.JointVel("dq_joint_head_roll", xml_name="joint_head_roll"),
            ObservationType.JointVel("dq_joint_head_yaw", xml_name="joint_head_yaw"),
            ObservationType.JointVel("dq_joint_left_shoulder_pitch", xml_name="joint_left_shoulder_pitch"),
            ObservationType.JointVel("dq_joint_left_shoulder_roll", xml_name="joint_left_shoulder_roll"),
            ObservationType.JointVel("dq_joint_left_shoulder_yaw", xml_name="joint_left_shoulder_yaw"),
            ObservationType.JointVel("dq_joint_left_elbow_pitch", xml_name="joint_left_elbow_pitch"),
            ObservationType.JointVel("dq_joint_left_wrist_yaw", xml_name="joint_left_wrist_yaw"),
            ObservationType.JointVel("dq_joint_left_wrist_roll", xml_name="joint_left_wrist_roll"),
            ObservationType.JointVel("dq_joint_left_wrist_pitch", xml_name="joint_left_wrist_pitch"),
            ObservationType.JointVel("dq_joint_right_shoulder_pitch", xml_name="joint_right_shoulder_pitch"),
            ObservationType.JointVel("dq_joint_right_shoulder_roll", xml_name="joint_right_shoulder_roll"),
            ObservationType.JointVel("dq_joint_right_shoulder_yaw", xml_name="joint_right_shoulder_yaw"),
            ObservationType.JointVel("dq_joint_right_elbow_pitch", xml_name="joint_right_elbow_pitch"),
            ObservationType.JointVel("dq_joint_right_wrist_yaw", xml_name="joint_right_wrist_yaw"),
            ObservationType.JointVel("dq_joint_right_wrist_roll", xml_name="joint_right_wrist_roll"),
            ObservationType.JointVel("dq_joint_right_wrist_pitch", xml_name="joint_right_wrist_pitch"),
        ]

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
            "link_left_hip_roll",
            "link_left_hip_yaw",
            "link_left_hip_pitch",
            "link_left_knee_pitch",
            "link_left_ankle_pitch",
            "link_left_ankle_roll",
            "link_right_hip_roll",
            "link_right_hip_yaw",
            "link_right_hip_pitch",
            "link_right_knee_pitch",
            "link_right_ankle_pitch",
            "link_right_ankle_roll",
            "link_waist_yaw",
            "link_waist_pitch",
            "link_waist_roll",
            "link_head_yaw",
            "link_head_roll",
            "link_head_pitch",
            "link_left_shoulder_pitch",
            "link_left_shoulder_roll",
            "link_left_shoulder_yaw",
            "link_left_elbow_pitch",
            "link_left_wrist_yaw",
            "link_left_wrist_roll",
            "link_left_wrist_pitch",
            "link_right_shoulder_pitch",
            "link_right_shoulder_roll",
            "link_right_shoulder_yaw",
            "link_right_elbow_pitch",
            "link_right_wrist_yaw",
            "link_right_wrist_roll",
            "link_right_wrist_pitch"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the XML file of the environment.

        Returns:
            str: Path to the default XML file.
        """
        return cls.get_model_path("fourier_gr1t2", "gr1t2.xml")

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco XML file.

        Returns:
            str: Name of the root body.
        """
        return "base"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.

        Returns:
            str: Name of the upper body.
        """
        return "link_torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root in the Mujoco XML file.

        Returns:
            str: Name of the root free joint.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when `HeightBasedTerminalStateHandler` is used.

        Returns:
            Tuple[float, float]: The healthy range of the root height.
        """
        return (0.6, 1.5)
