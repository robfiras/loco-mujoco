from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec
import numpy as np
import os

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core. utils import info_property


class UnitreeG1Inspire(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Unitree G1 robot with inspire hands.


    Default Observation Space TODO
    -----------------
    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 4        q_root                        FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_left_hip_pitch_joint        JointPos         [-2.5307]                            [2.8798]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_left_hip_roll_joint         JointPos         [-0.5236]                            [2.9671]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_hip_yaw_joint          JointPos         [-2.7576]                            [2.7576]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_knee_joint             JointPos         [-0.087267]                          [2.8798]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_ankle_pitch_joint      JointPos         [-0.87267]                           [0.5236]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_left_ankle_roll_joint       JointPos         [-0.2618]                            [0.2618]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_hip_pitch_joint       JointPos         [-2.5307]                            [2.8798]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_right_hip_roll_joint        JointPos         [-2.9671]                            [0.5236]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_hip_yaw_joint         JointPos         [-2.7576]                            [2.7576]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_knee_joint            JointPos         [-0.087267]                          [2.8798]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_ankle_pitch_joint     JointPos         [-0.87267]                           [0.5236]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_ankle_roll_joint      JointPos         [-0.2618]                            [0.2618]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_waist_yaw_joint             JointPos         [-2.618]                             [2.618]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_left_shoulder_pitch_joint   JointPos         [-3.0892]                            [2.6704]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_shoulder_roll_joint    JointPos         [-1.5882]                            [2.2515]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_shoulder_yaw_joint     JointPos         [-2.618]                             [2.618]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_left_elbow_joint            JointPos         [-1.0472]                            [2.0944]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_left_wrist_roll_joint       JointPos         [-1.97222]                           [1.97222]                      1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_right_shoulder_pitch_joint  JointPos         [-3.0892]                            [2.6704]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_right_shoulder_roll_joint   JointPos         [-2.2515]                            [1.5882]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_right_shoulder_yaw_joint    JointPos         [-2.618]                             [2.618]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_right_elbow_joint           JointPos         [-1.0472]                            [2.0944]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_right_wrist_roll_joint      JointPos         [-1.97222]                           [1.97222]                      1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28 - 33      dq_root                       FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_left_hip_pitch_joint       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           dq_left_hip_roll_joint        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           dq_left_hip_yaw_joint         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           dq_left_knee_joint            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           dq_left_ankle_pitch_joint     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           dq_left_ankle_roll_joint      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_right_hip_pitch_joint      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_right_hip_roll_joint       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_right_hip_yaw_joint        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_right_knee_joint           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_right_ankle_pitch_joint    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_right_ankle_roll_joint     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_waist_yaw_joint            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_left_shoulder_pitch_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_left_shoulder_roll_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_left_shoulder_yaw_joint    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_left_elbow_joint           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_left_wrist_roll_joint      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_right_shoulder_pitch_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_right_shoulder_roll_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_right_shoulder_yaw_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_right_elbow_joint          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_right_wrist_roll_joint     JointVel         [-inf]                               [inf]                          1
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    ----------------

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
    =============== ==== ===

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
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
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

                            # # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
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
        xml_file = "robot_models/scene_inspire_hands.xml"
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
