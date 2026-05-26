from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec
import numpy as np
import os

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core. utils import info_property


class UnitreeG1BallInspire(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Unitree G1 robot with inspire hands.


    Default Observation Space
    -----------------
    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 6        q_root                        FreeJointPosXY   [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      7
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_hip_pitch_joint        JointPos         -2.5307                              2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_hip_roll_joint         JointPos         -0.5236                              2.9671                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_hip_yaw_joint          JointPos         -2.7576                              2.7576                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_left_knee_joint             JointPos         -0.087267                            2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11          q_left_ankle_pitch_joint       JointPos         -0.87267                             0.5236                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_left_ankle_roll_joint       JointPos         -0.2618                              0.2618                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_hip_pitch_joint       JointPos         -2.5307                              2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_hip_roll_joint        JointPos         -2.9671                              0.5236                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_hip_yaw_joint         JointPos         -2.7576                              2.7576                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_knee_joint            JointPos         -0.087267                            2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_right_ankle_pitch_joint     JointPos         -0.87267                             0.5236                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_right_ankle_roll_joint      JointPos         -0.2618                              0.2618                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_waist_yaw_joint             JointPos         -2.618                               2.618                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_waist_roll_joint            JointPos         -0.52                                0.52                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_waist_pitch_joint           JointPos         -0.52                                0.52                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_left_shoulder_pitch_joint   JointPos         -3.0892                              2.6704                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_left_shoulder_roll_joint    JointPos         -1.5882                              2.2515                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_left_shoulder_yaw_joint     JointPos         -2.618                               2.618                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_left_elbow_joint            JointPos         -1.0472                              2.0944                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_left_wrist_roll_joint       JointPos         -1.97222                             1.97222                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_left_wrist_pitch_joint      JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_left_wrist_yaw_joint        JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_left_thumb_1_joint          JointPos         0                                    1.1641                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_left_thumb_2_joint          JointPos         0                                    0.5864                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_left_thumb_3_joint          JointPos         0                                    0.5                            1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_left_thumb_4_joint          JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_left_index_1_joint          JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_left_index_2_joint          JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           q_left_middle_1_joint         JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           q_left_middle_2_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           q_left_ring_1_joint           JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           q_left_ring_2_joint           JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           q_left_little_1_joint         JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           q_left_little_2_joint         JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           q_right_shoulder_pitch_joint  JointPos         -3.0892                              2.6704                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           q_right_shoulder_roll_joint   JointPos         -2.2515                              1.5882                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           q_right_shoulder_yaw_joint    JointPos         -2.618                               2.618                          1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           q_right_elbow_joint           JointPos         -1.0472                              2.0944                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           q_right_wrist_roll_joint      JointPos         -1.97222                             1.97222                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           q_right_wrist_pitch_joint     JointPos         -1.61443                             1.61443                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           q_right_wrist_yaw_joint       JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           q_right_thumb_1_joint         JointPos         0                                    1.1641                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           q_right_thumb_2_joint         JointPos         0                                    0.5864                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           q_right_thumb_3_joint         JointPos         0                                    0.5                            1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           q_right_thumb_4_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           q_right_index_1_joint         JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           q_right_index_2_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           q_right_middle_1_joint        JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           q_right_middle_2_joint        JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           q_right_ring_1_joint          JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           q_right_ring_2_joint          JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           q_right_little_1_joint        JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           q_right_little_2_joint        JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    60 - 66      q_ball_joint                  FreeJointPosXY   [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      7
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    ----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    ============ ============================= ================ ==================================== ============================== ===
    Index in Act Name                          Type             Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0            q_left_hip_pitch_joint        JointPos         -2.5307                              2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    1            q_left_hip_roll_joint         JointPos         -0.5236                              2.9671                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    2            q_left_hip_yaw_joint          JointPos         -2.7576                              2.7576                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    3            q_left_knee_joint             JointPos         -0.087267                            2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    4            q_left_ankle_pitch_joint       JointPos         -0.87267                             0.5236                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_left_ankle_roll_joint       JointPos         -0.2618                              0.2618                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_right_hip_pitch_joint       JointPos         -2.5307                              2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_right_hip_roll_joint        JointPos         -2.9671                              0.5236                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_right_hip_yaw_joint         JointPos         -2.7576                              2.7576                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_right_knee_joint            JointPos         -0.087267                            2.8798                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_right_ankle_pitch_joint     JointPos         -0.87267                             0.5236                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_ankle_roll_joint      JointPos         -0.2618                              0.2618                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_waist_yaw_joint             JointPos         -2.618                               2.618                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_waist_roll_joint            JointPos         -0.52                                0.52                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_waist_pitch_joint           JointPos         -0.52                                0.52                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_left_shoulder_pitch_joint   JointPos         -3.0892                              2.6704                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_left_shoulder_roll_joint    JointPos         -1.5882                              2.2515                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_left_shoulder_yaw_joint     JointPos         -2.618                               2.618                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_left_elbow_joint            JointPos         -1.0472                              2.0944                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_wrist_roll_joint       JointPos         -1.97222                             1.97222                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_wrist_pitch_joint      JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_left_wrist_yaw_joint        JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_left_thumb_1_joint          JointPos         0                                    1.1641                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_left_thumb_2_joint          JointPos         0                                    0.5864                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_left_thumb_3_joint          JointPos         0                                    0.5                            1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_left_thumb_4_joint          JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_left_index_1_joint          JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_left_index_2_joint          JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_left_middle_1_joint         JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_left_middle_2_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_left_ring_1_joint           JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_left_ring_2_joint           JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_left_little_1_joint         JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_left_little_2_joint         JointPos         0                                    3.14                           1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_right_shoulder_pitch_joint  JointPos         -3.0892                              2.6704                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           q_right_shoulder_roll_joint   JointPos         -2.2515                              1.5882                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           q_right_shoulder_yaw_joint    JointPos         -2.618                               2.618                          1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           q_right_elbow_joint           JointPos         -1.0472                              2.0944                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           q_right_wrist_roll_joint      JointPos         -1.97222                             1.97222                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           q_right_wrist_pitch_joint     JointPos         -1.61443                             1.61443                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           q_right_wrist_yaw_joint       JointPos         -1.61443                             1.61443                        1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           q_right_thumb_1_joint         JointPos         0                                    1.1641                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           q_right_thumb_2_joint         JointPos         0                                    0.5864                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           q_right_thumb_3_joint         JointPos         0                                    0.5                            1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           q_right_thumb_4_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           q_right_index_1_joint         JointPos         0                                    1.4381                         1 
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           q_right_index_2_joint         JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           q_right_middle_1_joint        JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           q_right_middle_2_joint        JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           q_right_ring_1_joint          JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           q_right_ring_2_joint          JointPos         0                                    3.14                           1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           q_right_little_1_joint        JointPos         0                                    1.4381                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           q_right_little_2_joint        JointPos         0                                    3.14                           1
    ============ ============================= ================ ==================================== ============================== ===

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_arms: bool = False,
                 disable_back_joint: bool = False,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_arms (bool): Whether to disable arm joints.
            disable_back_joint (bool): Whether to disable the back joint.
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

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
        if disable_arms or disable_back_joint:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] 
            # + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_xml_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, motors, and equality constraints should be removed from the Mujoco XML.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, motors to remove,
            and equality constraints to remove.
        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                                 "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                                 "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
                                 "right_index_1_joint", "right_index_2_joint", "right_middle_1_joint", "right_middle_2_joint",
                                 "right_ring_1_joint", "right_ring_2_joint", "right_little_1_joint", "right_little_2_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", 
                                 "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                                 "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",
                                 "left_index_1_joint", "left_index_2_joint", "left_middle_1_joint", "left_middle_2_joint",
                                 "left_ring_1_joint", "left_ring_2_joint", "left_little_1_joint", "left_little_2_joint"]
            motors_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                                 "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
                                 "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint",
                                 "right_index_1_joint", "right_index_2_joint", "right_middle_1_joint", "right_middle_2_joint",
                                 "right_ring_1_joint", "right_ring_2_joint", "right_little_1_joint", "right_little_2_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", 
                                 "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                                 "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",
                                 "left_index_1_joint", "left_index_2_joint", "left_middle_1_joint", "left_middle_2_joint",
                                 "left_ring_1_joint", "left_ring_2_joint", "left_little_1_joint", "left_little_2_joint"]
            

        if self._disable_back_joint:
            joints_to_remove += ["torso_joint"]
            motors_to_remove += ["torso_joint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: A list of observations.
        """

        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPos("q_root", xml_name="root"),
                            ObservationType.JointPos("q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointPos("q_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointPos("q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointPos("q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointPos("q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            ObservationType.JointPos("q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointPos("q_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointPos("q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointPos("q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointPos("q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            ObservationType.JointPos("q_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            ObservationType.JointPos("q_waist_roll_joint", xml_name="waist_roll_joint"),
                            ObservationType.JointPos("q_waist_pitch_joint", xml_name="waist_pitch_joint"),
                            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),

                            ObservationType.JointPos("q_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
                            ObservationType.JointPos("q_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
                            ObservationType.JointPos("q_left_thumb_1_joint", xml_name="left_thumb_1_joint"),
                            ObservationType.JointPos("q_left_thumb_2_joint", xml_name="left_thumb_2_joint"),
                            ObservationType.JointPos("q_left_thumb_3_joint", xml_name="left_thumb_3_joint"),
                            ObservationType.JointPos("q_left_thumb_4_joint", xml_name="left_thumb_4_joint"),
                            ObservationType.JointPos("q_left_index_1_joint", xml_name="left_index_1_joint"),
                            ObservationType.JointPos("q_left_index_2_joint", xml_name="left_index_2_joint"),
                            ObservationType.JointPos("q_left_middle_1_joint", xml_name="left_middle_1_joint"),
                            ObservationType.JointPos("q_left_middle_2_joint", xml_name="left_middle_2_joint"),
                            ObservationType.JointPos("q_left_ring_1_joint", xml_name="left_ring_1_joint"),
                            ObservationType.JointPos("q_left_ring_2_joint", xml_name="left_ring_2_joint"),
                            ObservationType.JointPos("q_left_little_1_joint", xml_name="left_little_1_joint"),
                            ObservationType.JointPos("q_left_little_2_joint", xml_name="left_little_2_joint"),

                            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),

                            ObservationType.JointPos("q_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
                            ObservationType.JointPos("q_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
                            ObservationType.JointPos("q_right_thumb_1_joint", xml_name="right_thumb_1_joint"),
                            ObservationType.JointPos("q_right_thumb_2_joint", xml_name="right_thumb_2_joint"),
                            ObservationType.JointPos("q_right_thumb_3_joint", xml_name="right_thumb_3_joint"),
                            ObservationType.JointPos("q_right_thumb_4_joint", xml_name="right_thumb_4_joint"),
                            ObservationType.JointPos("q_right_index_1_joint", xml_name="right_index_1_joint"),
                            ObservationType.JointPos("q_right_index_2_joint", xml_name="right_index_2_joint"),
                            ObservationType.JointPos("q_right_middle_1_joint", xml_name="right_middle_1_joint"),
                            ObservationType.JointPos("q_right_middle_2_joint", xml_name="right_middle_2_joint"),
                            ObservationType.JointPos("q_right_ring_1_joint", xml_name="right_ring_1_joint"),
                            ObservationType.JointPos("q_right_ring_2_joint", xml_name="right_ring_2_joint"),
                            ObservationType.JointPos("q_right_little_1_joint", xml_name="right_little_1_joint"),
                            ObservationType.JointPos("q_right_little_2_joint", xml_name="right_little_2_joint"),
                            ObservationType.FreeJointPos("q_ball_joint", xml_name="ball_joint"),

                            # # ------------- JOINT VEL -------------
                            # ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            # ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            # ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            # ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            # ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
                            # ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            # ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            # ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            # ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            # ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            # ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
                            # ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            # ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            # ObservationType.JointVel("dq_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            # ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            # ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            # ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            # ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
                            # ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            # ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            # ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            # ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            # ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
                            # ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),
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
        return [actuator.name for actuator in spec.actuators]

    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        """
        Reorients the arms to prevent collision with the hips when the arms are disabled.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        # modify the arm orientation
        left_shoulder_pitch_link = [body for body in spec.bodies if body.name == "left_shoulder_pitch_link"][0]
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = [body for body in spec.bodies if body.name == "right_elbow_link"][0]
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = [body for body in spec.bodies if body.name == "right_shoulder_pitch_link"][0]
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = [body for body in spec.bodies if body.name == "left_elbow_link"][0]
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the Unitree G1 environment.
        """
        xml_file = "robot_models/scene_g1_ball.xml"
        model_path = os.path.join(os.getcwd().removesuffix("/loco-mujoco/loco_mujoco/environments/humanoids"), xml_file)
        return model_path

    # @info_property
    # def root_free_joint_xml_name(self) -> str:
    #     """
    #     Returns the name of the root joint in the Mujoco XML file.
    #     """
    #     return "floating_base_joint"
    
    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.
        """
        return "torso_link"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height.

        Returns:
            Tuple[float, float]: The healthy height range (min, max).
        """
        return (0.5, 1.0)
