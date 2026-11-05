from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class ToddlerBot(BaseRobotHumanoid):

    """

    Description
    ------------


    Default Observation Space
    -------------------------
    ============ ========================== ================ ==================================== ============================== ===
    Index in Obs Name                       ObservationType  Min                                  Max                            Dim
    ============ ========================== ================ ==================================== ============================== ===
    0 - 4        q_root                     FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_neck_yaw_drive           JointPos         [-2.87979]                           [2.87979]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_neck_yaw_driven          JointPos         [-2.61799]                           [2.61799]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_neck_pitch               JointPos         [-1.39626]                           [0.610865]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_neck_pitch_act           JointPos         [-1.39626]                           [0.610865]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_waist_yaw                JointPos         [-1.5708]                            [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_waist_roll               JointPos         [-0.523599]                          [0.523599]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_waist_act_1              JointPos         [-4.67748]                           [4.67748]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_waist_act_2              JointPos         [-4.67748]                           [4.67748]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_left_hip_pitch           JointPos         [-1.5708]                            [2.35619]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_left_hip_roll            JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_left_hip_yaw_driven      JointPos         [-1.5708]                            [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_left_hip_yaw_drive       JointPos         [-1.8326]                            [1.8326]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_left_knee                JointPos         [-2.0944]                            [0.0]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_left_ank_pitch           JointPos         [-1.74533]                           [0.785398]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_ank_roll            JointPos         [-1.5708]                            [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_knee_act            JointPos         [-2.0944]                            [0.0]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_right_hip_pitch          JointPos         [-2.35619]                           [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_right_hip_roll           JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_right_hip_yaw_driven     JointPos         [-1.5708]                            [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_right_hip_yaw_drive      JointPos         [-1.8326]                            [1.8326]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_right_knee               JointPos         [0.0]                                [2.0944]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_right_ank_pitch          JointPos         [-0.785398]                          [1.74533]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_right_ank_roll           JointPos         [-1.5708]                            [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_right_knee_act           JointPos         [0.0]                                [2.0944]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_left_sho_pitch           JointPos         [-3.14159]                           [1.5708]                       1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_left_sho_roll            JointPos         [-1.5708]                            [0.349066]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_left_sho_yaw_drive       JointPos         [-2.61799]                           [2.61799]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_left_elbow_roll          JointPos         [-1.91986]                           [2.44346]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_left_elbow_yaw_drive     JointPos         [-2.61799]                           [2.61799]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_left_wrist_pitch_drive   JointPos         [-1.91986]                           [1.39626]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           q_left_wrist_roll          JointPos         [-1.91986]                           [1.39626]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           q_right_sho_pitch          JointPos         [-1.5708]                            [3.14159]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           q_right_sho_roll           JointPos         [-1.5708]                            [0.349066]                     1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           q_right_sho_yaw_drive      JointPos         [-2.61799]                           [2.61799]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           q_right_elbow_roll         JointPos         [-1.91986]                           [2.44346]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           q_right_elbow_yaw_drive    JointPos         [-2.61799]                           [2.61799]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           q_right_wrist_pitch_drive  JointPos         [-1.39626]                           [1.91986]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           q_right_wrist_roll         JointPos         [-1.39626]                           [1.91986]                      1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43 - 48      dq_root                    FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_neck_yaw_drive          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_neck_yaw_driven         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_neck_pitch              JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_neck_pitch_act          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_waist_yaw               JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_waist_roll              JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_waist_act_1             JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_waist_act_2             JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_left_hip_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_left_hip_roll           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_left_hip_yaw_driven     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_left_hip_yaw_drive      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    61           dq_left_knee               JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    62           dq_left_ank_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    63           dq_left_ank_roll           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    64           dq_left_knee_act           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    65           dq_right_hip_pitch         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    66           dq_right_hip_roll          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    67           dq_right_hip_yaw_driven    JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    68           dq_right_hip_yaw_drive     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    69           dq_right_knee              JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    70           dq_right_ank_pitch         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    71           dq_right_ank_roll          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    72           dq_right_knee_act          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    73           dq_left_sho_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    74           dq_left_sho_roll           JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    75           dq_left_sho_yaw_drive      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    76           dq_left_elbow_roll         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    77           dq_left_elbow_yaw_drive    JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    78           dq_left_wrist_pitch_drive  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    79           dq_left_wrist_roll         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    80           dq_right_sho_pitch         JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    81           dq_right_sho_roll          JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    82           dq_right_sho_yaw_drive     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    83           dq_right_elbow_roll        JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    84           dq_right_elbow_yaw_drive   JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    85           dq_right_wrist_pitch_drive JointVel         [-inf]                               [inf]                          1
    ------------ -------------------------- ---------------- ------------------------------------ ------------------------------ ---
    86           dq_right_wrist_roll        JointVel         [-inf]                               [inf]                          1
    ============ ========================== ================ ==================================== ============================== ===



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
    =============== ==== ===

    Methods
    ---------

    """

    mjx_enabled = False

    def __init__(self, spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Initializes the ToddlerBot environment.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
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
            List[Observation]: A list of observations.
        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_neck_yaw_drive", xml_name="neck_yaw_drive"),
            ObservationType.JointPos("q_neck_yaw_driven", xml_name="neck_yaw_driven"),
            ObservationType.JointPos("q_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointPos("q_neck_pitch_act", xml_name="neck_pitch_act"),
            ObservationType.JointPos("q_waist_yaw", xml_name="waist_yaw"),
            ObservationType.JointPos("q_waist_roll", xml_name="waist_roll"),
            ObservationType.JointPos("q_waist_act_1", xml_name="waist_act_1"),
            ObservationType.JointPos("q_waist_act_2", xml_name="waist_act_2"),
            ObservationType.JointPos("q_left_hip_pitch", xml_name="left_hip_pitch"),
            ObservationType.JointPos("q_left_hip_roll", xml_name="left_hip_roll"),
            ObservationType.JointPos("q_left_hip_yaw_driven", xml_name="left_hip_yaw_driven"),
            ObservationType.JointPos("q_left_hip_yaw_drive", xml_name="left_hip_yaw_drive"),
            ObservationType.JointPos("q_left_knee", xml_name="left_knee"),
            ObservationType.JointPos("q_left_ank_pitch", xml_name="left_ank_pitch"),
            ObservationType.JointPos("q_left_ank_roll", xml_name="left_ank_roll"),
            ObservationType.JointPos("q_left_knee_act", xml_name="left_knee_act"),
            ObservationType.JointPos("q_right_hip_pitch", xml_name="right_hip_pitch"),
            ObservationType.JointPos("q_right_hip_roll", xml_name="right_hip_roll"),
            ObservationType.JointPos("q_right_hip_yaw_driven", xml_name="right_hip_yaw_driven"),
            ObservationType.JointPos("q_right_hip_yaw_drive", xml_name="right_hip_yaw_drive"),
            ObservationType.JointPos("q_right_knee", xml_name="right_knee"),
            ObservationType.JointPos("q_right_ank_pitch", xml_name="right_ank_pitch"),
            ObservationType.JointPos("q_right_ank_roll", xml_name="right_ank_roll"),
            ObservationType.JointPos("q_right_knee_act", xml_name="right_knee_act"),
            ObservationType.JointPos("q_left_sho_pitch", xml_name="left_sho_pitch"),
            ObservationType.JointPos("q_left_sho_roll", xml_name="left_sho_roll"),
            ObservationType.JointPos("q_left_sho_yaw_drive", xml_name="left_sho_yaw_drive"),
            ObservationType.JointPos("q_left_elbow_roll", xml_name="left_elbow_roll"),
            ObservationType.JointPos("q_left_elbow_yaw_drive", xml_name="left_elbow_yaw_drive"),
            ObservationType.JointPos("q_left_wrist_pitch_drive", xml_name="left_wrist_pitch_drive"),
            ObservationType.JointPos("q_left_wrist_roll", xml_name="left_wrist_roll"),
            ObservationType.JointPos("q_right_sho_pitch", xml_name="right_sho_pitch"),
            ObservationType.JointPos("q_right_sho_roll", xml_name="right_sho_roll"),
            ObservationType.JointPos("q_right_sho_yaw_drive", xml_name="right_sho_yaw_drive"),
            ObservationType.JointPos("q_right_elbow_roll", xml_name="right_elbow_roll"),
            ObservationType.JointPos("q_right_elbow_yaw_drive", xml_name="right_elbow_yaw_drive"),
            ObservationType.JointPos("q_right_wrist_pitch_drive", xml_name="right_wrist_pitch_drive"),
            ObservationType.JointPos("q_right_wrist_roll", xml_name="right_wrist_roll"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_neck_yaw_drive", xml_name="neck_yaw_drive"),
            ObservationType.JointVel("dq_neck_yaw_driven", xml_name="neck_yaw_driven"),
            ObservationType.JointVel("dq_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointVel("dq_neck_pitch_act", xml_name="neck_pitch_act"),
            ObservationType.JointVel("dq_waist_yaw", xml_name="waist_yaw"),
            ObservationType.JointVel("dq_waist_roll", xml_name="waist_roll"),
            ObservationType.JointVel("dq_waist_act_1", xml_name="waist_act_1"),
            ObservationType.JointVel("dq_waist_act_2", xml_name="waist_act_2"),
            ObservationType.JointVel("dq_left_hip_pitch", xml_name="left_hip_pitch"),
            ObservationType.JointVel("dq_left_hip_roll", xml_name="left_hip_roll"),
            ObservationType.JointVel("dq_left_hip_yaw_driven", xml_name="left_hip_yaw_driven"),
            ObservationType.JointVel("dq_left_hip_yaw_drive", xml_name="left_hip_yaw_drive"),
            ObservationType.JointVel("dq_left_knee", xml_name="left_knee"),
            ObservationType.JointVel("dq_left_ank_pitch", xml_name="left_ank_pitch"),
            ObservationType.JointVel("dq_left_ank_roll", xml_name="left_ank_roll"),
            ObservationType.JointVel("dq_left_knee_act", xml_name="left_knee_act"),
            ObservationType.JointVel("dq_right_hip_pitch", xml_name="right_hip_pitch"),
            ObservationType.JointVel("dq_right_hip_roll", xml_name="right_hip_roll"),
            ObservationType.JointVel("dq_right_hip_yaw_driven", xml_name="right_hip_yaw_driven"),
            ObservationType.JointVel("dq_right_hip_yaw_drive", xml_name="right_hip_yaw_drive"),
            ObservationType.JointVel("dq_right_knee", xml_name="right_knee"),
            ObservationType.JointVel("dq_right_ank_pitch", xml_name="right_ank_pitch"),
            ObservationType.JointVel("dq_right_ank_roll", xml_name="right_ank_roll"),
            ObservationType.JointVel("dq_right_knee_act", xml_name="right_knee_act"),
            ObservationType.JointVel("dq_left_sho_pitch", xml_name="left_sho_pitch"),
            ObservationType.JointVel("dq_left_sho_roll", xml_name="left_sho_roll"),
            ObservationType.JointVel("dq_left_sho_yaw_drive", xml_name="left_sho_yaw_drive"),
            ObservationType.JointVel("dq_left_elbow_roll", xml_name="left_elbow_roll"),
            ObservationType.JointVel("dq_left_elbow_yaw_drive", xml_name="left_elbow_yaw_drive"),
            ObservationType.JointVel("dq_left_wrist_pitch_drive", xml_name="left_wrist_pitch_drive"),
            ObservationType.JointVel("dq_left_wrist_roll", xml_name="left_wrist_roll"),
            ObservationType.JointVel("dq_right_sho_pitch", xml_name="right_sho_pitch"),
            ObservationType.JointVel("dq_right_sho_roll", xml_name="right_sho_roll"),
            ObservationType.JointVel("dq_right_sho_yaw_drive", xml_name="right_sho_yaw_drive"),
            ObservationType.JointVel("dq_right_elbow_roll", xml_name="right_elbow_roll"),
            ObservationType.JointVel("dq_right_elbow_yaw_drive", xml_name="right_elbow_yaw_drive"),
            ObservationType.JointVel("dq_right_wrist_pitch_drive", xml_name="right_wrist_pitch_drive"),
            ObservationType.JointVel("dq_right_wrist_roll", xml_name="right_wrist_roll"),
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Returns the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """

        action_spec = [
            "neck_yaw_drive",
            "neck_pitch_act",
            "waist_act_1",
            "waist_act_2",
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw_drive",
            "left_knee_act",
            "left_ank_roll",
            "left_ank_pitch",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw_drive",
            "right_knee_act",
            "right_ank_roll",
            "right_ank_pitch",
            "left_sho_pitch",
            "left_sho_roll",
            "left_sho_yaw_drive",
            "left_elbow_roll",
            "left_elbow_yaw_drive",
            "left_wrist_pitch_drive",
            "left_wrist_roll",
            "right_sho_pitch",
            "right_sho_roll",
            "right_sho_yaw_drive",
            "right_elbow_roll",
            "right_elbow_yaw_drive",
            "right_wrist_pitch_drive",
            "right_wrist_roll"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the ToddlerBot environment.
        """
        return cls.get_model_path("toddlerbot", "toddlerbot.xml")

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body specified in the Mujoco XML file.

        """
        return "torso"

    @info_property
    def upper_body_xml_name(self) -> str:
        """ Returns the name of the upper body in the Mujoco XML file. """
        return "spur_1m_20t"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """ Returns the name of the free joint in the Mujoco XML file. """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height.

        Returns:
            Tuple[float, float]: The healthy height range (min, max).
        """
        return (0.2, 0.5)
