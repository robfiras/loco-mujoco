from typing import Union, List, Dict, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class UnitreeH1v2(BaseRobotHumanoid):

    """

    Description
    ------------
    Mujoco environment of the Unitree H1.5 robot.

    Default Observation Space
    -------------------------

    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 4        q_floating_base_joint         FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_left_hip_yaw_joint          JointPos         [-0.43]                              [0.43]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_left_hip_pitch_joint        JointPos         [-3.14]                              [2.5]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_hip_roll_joint         JointPos         [-0.43]                              [3.14]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_knee_joint             JointPos         [-0.12]                              [2.19]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_ankle_pitch_joint      JointPos         [-0.897334]                          [0.523598]                     1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_left_ankle_roll_joint       JointPos         [-0.261799]                          [0.261799]                     1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_hip_yaw_joint         JointPos         [-0.43]                              [0.43]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_right_hip_pitch_joint       JointPos         [-3.14]                              [2.5]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_hip_roll_joint        JointPos         [-3.14]                              [0.43]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_knee_joint            JointPos         [-0.12]                              [2.19]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_ankle_pitch_joint     JointPos         [-0.897334]                          [0.523598]                     1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_ankle_roll_joint      JointPos         [-0.261799]                          [0.261799]                     1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_torso_joint                 JointPos         [-2.35]                              [2.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_left_shoulder_pitch_joint   JointPos         [-3.14]                              [1.57]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_shoulder_roll_joint    JointPos         [-0.38]                              [3.4]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_shoulder_yaw_joint     JointPos         [-2.66]                              [3.01]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_left_elbow_joint            JointPos         [-0.95]                              [3.18]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_right_shoulder_pitch_joint  JointPos         [-3.14]                              [1.57]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_right_shoulder_roll_joint   JointPos         [-3.4]                               [0.38]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_right_shoulder_yaw_joint    JointPos         [-3.01]                              [2.66]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_right_elbow_joint           JointPos         [-0.95]                              [3.18]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26 - 31      dq_floating_base_joint        FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           dq_left_hip_yaw_joint         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           dq_left_hip_pitch_joint       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_left_hip_roll_joint        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           dq_left_knee_joint            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           dq_left_ankle_pitch_joint     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           dq_left_ankle_roll_joint      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           dq_right_hip_yaw_joint        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           dq_right_hip_pitch_joint      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_right_hip_roll_joint       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_right_knee_joint           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_right_ankle_pitch_joint    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_right_ankle_roll_joint     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_torso_joint                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_left_shoulder_pitch_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_left_shoulder_roll_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_left_shoulder_yaw_joint    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_left_elbow_joint           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_right_shoulder_pitch_joint JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_right_shoulder_roll_joint  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_right_shoulder_yaw_joint   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_right_elbow_joint          JointVel         [-inf]                               [inf]                          1
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    --------------------

    Control function type: **PDControl**

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
    =============== ==== ===


    Methods
    ---------

    """

    mjx_enabled = False

    def __init__(self, disable_hands: bool = True, spec: Union[str, MjSpec] = None,
                 observation_spec: List[ObservationType] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_hands (bool): Whether to disable hand joints.
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[ObservationType], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        self._disable_handss = disable_hands

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
        if disable_hands:
            joints_to_remove, actuators_to_remove, equ_constraints_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in actuators_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          actuators_to_remove, equ_constraints_to_remove)

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=[self.p_gains[act.name] for act in spec.actuators],
                                            d_gain=[self.d_gains[act.name] for act in spec.actuators],
                                            scale_action_to_jnt_limits=False)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = [
            # Left Hand
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        actuators_to_remove = [
            # Left Hand
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            # Right Hand
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
        ]

        equ_constr_to_remove = []

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[ObservationType]: A list of observation types.
        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_floating_base_joint", xml_name="floating_base_joint"),
            ObservationType.JointPos("q_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
            ObservationType.JointPos("q_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
            ObservationType.JointPos("q_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
            ObservationType.JointPos("q_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointPos("q_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
            ObservationType.JointPos("q_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
            ObservationType.JointPos("q_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
            ObservationType.JointPos("q_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
            ObservationType.JointPos("q_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
            ObservationType.JointPos("q_right_knee_joint", xml_name="right_knee_joint"),
            ObservationType.JointPos("q_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
            ObservationType.JointPos("q_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
            ObservationType.JointPos("q_torso_joint", xml_name="torso_joint"),
            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
            ObservationType.JointPos("q_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
            ObservationType.JointPos("q_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),
            ObservationType.JointPos("q_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
            ObservationType.JointPos("q_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
            ObservationType.JointPos("q_L_index_proximal_joint", xml_name="L_index_proximal_joint"),
            ObservationType.JointPos("q_L_index_intermediate_joint", xml_name="L_index_intermediate_joint"),
            ObservationType.JointPos("q_L_middle_proximal_joint", xml_name="L_middle_proximal_joint"),
            ObservationType.JointPos("q_L_middle_intermediate_joint", xml_name="L_middle_intermediate_joint"),
            ObservationType.JointPos("q_L_ring_proximal_joint", xml_name="L_ring_proximal_joint"),
            ObservationType.JointPos("q_L_ring_intermediate_joint", xml_name="L_ring_intermediate_joint"),
            ObservationType.JointPos("q_L_pinky_proximal_joint", xml_name="L_pinky_proximal_joint"),
            ObservationType.JointPos("q_L_pinky_intermediate_joint", xml_name="L_pinky_intermediate_joint"),
            ObservationType.JointPos("q_L_thumb_proximal_yaw_joint", xml_name="L_thumb_proximal_yaw_joint"),
            ObservationType.JointPos("q_L_thumb_proximal_pitch_joint", xml_name="L_thumb_proximal_pitch_joint"),
            ObservationType.JointPos("q_L_thumb_intermediate_joint", xml_name="L_thumb_intermediate_joint"),
            ObservationType.JointPos("q_L_thumb_distal_joint", xml_name="L_thumb_distal_joint"),
            ObservationType.JointPos("q_R_index_proximal_joint", xml_name="R_index_proximal_joint"),
            ObservationType.JointPos("q_R_index_intermediate_joint", xml_name="R_index_intermediate_joint"),
            ObservationType.JointPos("q_R_middle_proximal_joint", xml_name="R_middle_proximal_joint"),
            ObservationType.JointPos("q_R_middle_intermediate_joint", xml_name="R_middle_intermediate_joint"),
            ObservationType.JointPos("q_R_ring_proximal_joint", xml_name="R_ring_proximal_joint"),
            ObservationType.JointPos("q_R_ring_intermediate_joint", xml_name="R_ring_intermediate_joint"),
            ObservationType.JointPos("q_R_pinky_proximal_joint", xml_name="R_pinky_proximal_joint"),
            ObservationType.JointPos("q_R_pinky_intermediate_joint", xml_name="R_pinky_intermediate_joint"),
            ObservationType.JointPos("q_R_thumb_proximal_yaw_joint", xml_name="R_thumb_proximal_yaw_joint"),
            ObservationType.JointPos("q_R_thumb_proximal_pitch_joint", xml_name="R_thumb_proximal_pitch_joint"),
            ObservationType.JointPos("q_R_thumb_intermediate_joint", xml_name="R_thumb_intermediate_joint"),
            ObservationType.JointPos("q_R_thumb_distal_joint", xml_name="R_thumb_distal_joint"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_floating_base_joint", xml_name="floating_base_joint"),
            ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
            ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
            ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
            ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
            ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
            ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
            ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
            ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
            ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
            ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
            ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
            ObservationType.JointVel("dq_torso_joint", xml_name="torso_joint"),
            ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
            ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
            ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
            ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
            ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
            ObservationType.JointVel("dq_left_wrist_pitch_joint", xml_name="left_wrist_pitch_joint"),
            ObservationType.JointVel("dq_left_wrist_yaw_joint", xml_name="left_wrist_yaw_joint"),
            ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
            ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
            ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
            ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
            ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),
            ObservationType.JointVel("dq_right_wrist_pitch_joint", xml_name="right_wrist_pitch_joint"),
            ObservationType.JointVel("dq_right_wrist_yaw_joint", xml_name="right_wrist_yaw_joint"),
            ObservationType.JointVel("dq_L_index_proximal_joint", xml_name="L_index_proximal_joint"),
            ObservationType.JointVel("dq_L_index_intermediate_joint", xml_name="L_index_intermediate_joint"),
            ObservationType.JointVel("dq_L_middle_proximal_joint", xml_name="L_middle_proximal_joint"),
            ObservationType.JointVel("dq_L_middle_intermediate_joint", xml_name="L_middle_intermediate_joint"),
            ObservationType.JointVel("dq_L_ring_proximal_joint", xml_name="L_ring_proximal_joint"),
            ObservationType.JointVel("dq_L_ring_intermediate_joint", xml_name="L_ring_intermediate_joint"),
            ObservationType.JointVel("dq_L_pinky_proximal_joint", xml_name="L_pinky_proximal_joint"),
            ObservationType.JointVel("dq_L_pinky_intermediate_joint", xml_name="L_pinky_intermediate_joint"),
            ObservationType.JointVel("dq_L_thumb_proximal_yaw_joint", xml_name="L_thumb_proximal_yaw_joint"),
            ObservationType.JointVel("dq_L_thumb_proximal_pitch_joint", xml_name="L_thumb_proximal_pitch_joint"),
            ObservationType.JointVel("dq_L_thumb_intermediate_joint", xml_name="L_thumb_intermediate_joint"),
            ObservationType.JointVel("dq_L_thumb_distal_joint", xml_name="L_thumb_distal_joint"),
            ObservationType.JointVel("dq_R_index_proximal_joint", xml_name="R_index_proximal_joint"),
            ObservationType.JointVel("dq_R_index_intermediate_joint", xml_name="R_index_intermediate_joint"),
            ObservationType.JointVel("dq_R_middle_proximal_joint", xml_name="R_middle_proximal_joint"),
            ObservationType.JointVel("dq_R_middle_intermediate_joint", xml_name="R_middle_intermediate_joint"),
            ObservationType.JointVel("dq_R_ring_proximal_joint", xml_name="R_ring_proximal_joint"),
            ObservationType.JointVel("dq_R_ring_intermediate_joint", xml_name="R_ring_intermediate_joint"),
            ObservationType.JointVel("dq_R_pinky_proximal_joint", xml_name="R_pinky_proximal_joint"),
            ObservationType.JointVel("dq_R_pinky_intermediate_joint", xml_name="R_pinky_intermediate_joint"),
            ObservationType.JointVel("dq_R_thumb_proximal_yaw_joint", xml_name="R_thumb_proximal_yaw_joint"),
            ObservationType.JointVel("dq_R_thumb_proximal_pitch_joint", xml_name="R_thumb_proximal_pitch_joint"),
            ObservationType.JointVel("dq_R_thumb_intermediate_joint", xml_name="R_thumb_intermediate_joint"),
            ObservationType.JointVel("dq_R_thumb_distal_joint", xml_name="R_thumb_distal_joint"),
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
        ...
        action_spec = [
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint"
        ]

        return action_spec

    @property
    def p_gains(self) -> Dict[str, float]:
        """
        Getter for the proportional gains of the PD controller.

        Returns:
            Dict[str, float]: A dictionary containing the proportional gains for each joint

        """
        p_gains = {
            'left_hip_yaw_joint': 200.0,
            'left_hip_pitch_joint': 200.0,
            'left_hip_roll_joint': 200.0,
            'left_knee_joint': 300.0,
            'left_ankle_pitch_joint': 40.0,
            'left_ankle_roll_joint': 40.0,
            'right_hip_yaw_joint': 200.0,
            'right_hip_pitch_joint': 200.0,
            'right_hip_roll_joint': 200.0,
            'right_knee_joint': 300.0,
            'right_ankle_pitch_joint': 40.0,
            'right_ankle_roll_joint': 40.0,
            'torso_joint': 200.0,
            'left_shoulder_pitch_joint': 40.0,
            'left_shoulder_roll_joint': 40.0,
            'left_shoulder_yaw_joint': 18.0,
            'left_elbow_joint': 18.0,
            'left_wrist_roll_joint': 19.0,
            'left_wrist_pitch_joint': 19.0,
            'left_wrist_yaw_joint': 19.0,
            'right_shoulder_pitch_joint': 40.0,
            'right_shoulder_roll_joint': 40.0,
            'right_shoulder_yaw_joint': 18.0,
            'right_elbow_joint': 18.0,
            'right_wrist_roll_joint': 19.0,
            'right_wrist_pitch_joint': 19.0,
            'right_wrist_yaw_joint': 19.0,
            'L_index_proximal_joint': 1.0,
            'L_index_intermediate_joint': 1.0,
            'L_middle_proximal_joint': 1.0,
            'L_middle_intermediate_joint': 1.0,
            'L_ring_proximal_joint': 1.0,
            'L_ring_intermediate_joint': 1.0,
            'L_pinky_proximal_joint': 1.0,
            'L_pinky_intermediate_joint': 1.0,
            'L_thumb_proximal_yaw_joint': 1.0,
            'L_thumb_proximal_pitch_joint': 1.0,
            'L_thumb_intermediate_joint': 1.0,
            'L_thumb_distal_joint': 1.0,
            'R_index_proximal_joint': 1.0,
            'R_index_intermediate_joint': 1.0,
            'R_middle_proximal_joint': 1.0,
            'R_middle_intermediate_joint': 1.0,
            'R_ring_proximal_joint': 1.0,
            'R_ring_intermediate_joint': 1.0,
            'R_pinky_proximal_joint': 1.0,
            'R_pinky_intermediate_joint': 1.0,
            'R_thumb_proximal_yaw_joint': 1.0,
            'R_thumb_proximal_pitch_joint': 1.0,
            'R_thumb_intermediate_joint': 1.0,
            'R_thumb_distal_joint': 1.0,
        }

        return p_gains

    @property
    def d_gains(self) -> Dict[str, float]:
        """
        Getter for the derivative gains of the PD controller.

        Returns:
            Dict[str, float]: A dictionary containing the derivative gains for each joint

        """
        d_gains = {
            'left_hip_yaw_joint': 2.5,
            'left_hip_pitch_joint': 2.5,
            'left_hip_roll_joint': 2.5,
            'left_knee_joint': 4.0,
            'left_ankle_pitch_joint': 2.0,
            'left_ankle_roll_joint': 2.0,
            'right_hip_yaw_joint': 2.5,
            'right_hip_pitch_joint': 2.5,
            'right_hip_roll_joint': 2.5,
            'right_knee_joint': 4.0,
            'right_ankle_pitch_joint': 2.0,
            'right_ankle_roll_joint': 2.0,
            'torso_joint': 2.5,
            'left_shoulder_pitch_joint': 2.0,
            'left_shoulder_roll_joint': 2.0,
            'left_shoulder_yaw_joint': 1.8,
            'left_elbow_joint': 1.8,
            'left_wrist_roll_joint': 1.9,
            'left_wrist_pitch_joint': 1.9,
            'left_wrist_yaw_joint': 1.9,
            'right_shoulder_pitch_joint': 2.0,
            'right_shoulder_roll_joint': 2.0,
            'right_shoulder_yaw_joint': 1.8,
            'right_elbow_joint': 1.8,
            'right_wrist_roll_joint': 1.9,
            'right_wrist_pitch_joint': 1.9,
            'right_wrist_yaw_joint': 1.9,
            'L_index_proximal_joint': 0.1,
            'L_index_intermediate_joint': 0.1,
            'L_middle_proximal_joint': 0.1,
            'L_middle_intermediate_joint': 0.1,
            'L_ring_proximal_joint': 0.1,
            'L_ring_intermediate_joint': 0.1,
            'L_pinky_proximal_joint': 0.1,
            'L_pinky_intermediate_joint': 0.1,
            'L_thumb_proximal_yaw_joint': 0.1,
            'L_thumb_proximal_pitch_joint': 0.1,
            'L_thumb_intermediate_joint': 0.1,
            'L_thumb_distal_joint': 0.1,
            'R_index_proximal_joint': 0.1,
            'R_index_intermediate_joint': 0.1,
            'R_middle_proximal_joint': 0.1,
            'R_middle_intermediate_joint': 0.1,
            'R_ring_proximal_joint': 0.1,
            'R_ring_intermediate_joint': 0.1,
            'R_pinky_proximal_joint': 0.1,
            'R_pinky_intermediate_joint': 0.1,
            'R_thumb_proximal_yaw_joint': 0.1,
            'R_thumb_proximal_pitch_joint': 0.1,
            'R_thumb_intermediate_joint': 0.1,
            'R_thumb_distal_joint': 0.1,
        }

        return d_gains

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the unitree h1_2 environment.
        """
        return cls.get_model_path("unitree_h1_2", "h1_2.xml")

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body XML element.

        """
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint XML element.

        """
        return "floating_base_joint"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        """
        return (0.6, 1.5)
