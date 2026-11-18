"""
Environment of the Talos humanoid robot (CPU MuJoCo backend).
"""

from typing import Union, List, Tuple, Dict

import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.core.utils import info_property


class Talos(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Talos robot. Talos is a humanoid robot developed by PAL Robotics.

    The robot has a floating base, 2 torso joints, 2 head joints, 7 joints per arm, 6 joints per leg and two
    grippers. By default the grippers are removed (``disable_gripper=True``), which leaves 30 actuated
    joints. The default observation then consists of the pose of the floating base without its x-y position
    (height and orientation quaternion), all joint positions, the linear and angular velocity of the floating
    base, and all joint velocities. The action contains one command per actuator, normalized to [-1, 1] and
    rescaled to the control range of the actuator (default control function).

    With ``disable_gripper=False``, the gripper joints are added to the observation and one actuator per
    gripper is added to the action space. The tables below show the default spaces without grippers.


    Default Observation Space
    -------------------------

    ============ ==================== ================ ==================================== ============================== ===
    Index in Obs Name                 ObservationType  Min                                  Max                            Dim
    ============ ==================== ================ ==================================== ============================== ===
    0 - 4        q_reference          FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_torso_1_joint      JointPos         [-1.25664]                           [1.25664]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_torso_2_joint      JointPos         [-0.226893]                          [0.733038]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_head_1_joint       JointPos         [-0.20944]                           [0.785398]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_head_2_joint       JointPos         [-1.309]                             [1.309]                        1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_arm_left_1_joint   JointPos         [-1.5708]                            [0.785398]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_arm_left_2_joint   JointPos         [0.00872665]                         [2.87107]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_arm_left_3_joint   JointPos         [-2.42601]                           [2.42601]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_arm_left_4_joint   JointPos         [-2.23402]                           [-0.00349066]                  1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_arm_left_5_joint   JointPos         [-2.51327]                           [2.51327]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_arm_left_6_joint   JointPos         [-1.37008]                           [1.37008]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_arm_left_7_joint   JointPos         [-0.680678]                          [0.680678]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_arm_right_1_joint  JointPos         [-0.785398]                          [1.5708]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_arm_right_2_joint  JointPos         [-2.87107]                           [-0.00872665]                  1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_arm_right_3_joint  JointPos         [-2.42601]                           [2.42601]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_arm_right_4_joint  JointPos         [-2.23402]                           [-0.00349066]                  1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_arm_right_5_joint  JointPos         [-2.51327]                           [2.51327]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_arm_right_6_joint  JointPos         [-1.37008]                           [1.37008]                      1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_arm_right_7_joint  JointPos         [-0.680678]                          [0.680678]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_leg_left_1_joint   JointPos         [-0.349066]                          [1.5708]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_leg_left_2_joint   JointPos         [-0.5236]                            [0.5236]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_leg_left_3_joint   JointPos         [-2.095]                             [0.7]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_leg_left_4_joint   JointPos         [0.0]                                [2.618]                        1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_leg_left_5_joint   JointPos         [-1.27]                              [0.68]                         1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_leg_left_6_joint   JointPos         [-0.5236]                            [0.5236]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_leg_right_1_joint  JointPos         [-1.5708]                            [0.349066]                     1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_leg_right_2_joint  JointPos         [-0.5236]                            [0.5236]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_leg_right_3_joint  JointPos         [-2.095]                             [0.7]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_leg_right_4_joint  JointPos         [0.0]                                [2.618]                        1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_leg_right_5_joint  JointPos         [-1.27]                              [0.68]                         1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_leg_right_6_joint  JointPos         [-0.5236]                            [0.5236]                       1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    35 - 40      dq_reference         FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_torso_1_joint     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_torso_2_joint     JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_head_1_joint      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_head_2_joint      JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_arm_left_1_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_arm_left_2_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_arm_left_3_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_arm_left_4_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_arm_left_5_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_arm_left_6_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_arm_left_7_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_arm_right_1_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_arm_right_2_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_arm_right_3_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_arm_right_4_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_arm_right_5_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_arm_right_6_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_arm_right_7_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_leg_left_1_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_leg_left_2_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    61           dq_leg_left_3_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    62           dq_leg_left_4_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    63           dq_leg_left_5_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    64           dq_leg_left_6_joint  JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    65           dq_leg_right_1_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    66           dq_leg_right_2_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    67           dq_leg_right_3_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    68           dq_leg_right_4_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    69           dq_leg_right_5_joint JointVel         [-inf]                               [inf]                          1
    ------------ -------------------- ---------------- ------------------------------------ ------------------------------ ---
    70           dq_leg_right_6_joint JointVel         [-inf]                               [inf]                          1
    ============ ==================== ================ ==================================== ============================== ===

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
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_gripper: bool = True,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None, **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_gripper (bool): If True, the joints, actuators and equality constraints of both grippers
                are removed from the model, and the respective entries are removed from the observation and
                action specification.
            spec (Union[str, MjSpec]): Model to simulate, either a path to a MuJoCo XML file or an ``MjSpec``
                object. If None, the default model of this environment is used.
            observation_spec (List[Observation]): Observations that make up the observation vector, given as
                observation objects or as dictionaries with a ``type`` key naming the observation type and its
                parameters as the remaining keys. If None, the default observation space is used.
            actuation_spec (List[str]): Names of the actuators in the model that are controlled by the action.
                If None, the default action space is used.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.base.LocoEnv`. The
                extra key ``mjx_reduced_contacts`` (bool, default True) only affects the Mjx variant: if True,
                the contacts are reduced by ``_modify_spec_for_mjx``; if False, the collision geometries of the
                model are kept.

        """

        self._disable_gripper = disable_gripper

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

        if disable_gripper:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]

            spec = self._delete_from_spec(spec, joints_to_remove, motors_to_remove, equ_constr_to_remove)

        # mjx_reduced_contacts=False keeps the robot's own collision geoms on Mjx/MjWarp instead of the
        # simplified feet-only contacts of _modify_spec_for_mjx
        if kwargs.pop("mjx_reduced_contacts", True) and self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, motors, and equality constraints should be removed from the Mujoco spec.

        Returns:
            Tuple[List[str], List[str], List[str]]: Names of the joints to remove, names of the motors to
            remove, and names of the equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_gripper:
            joints_to_remove += [
                "gripper_left_joint",
                "gripper_left_inner_double_joint",
                "gripper_left_motor_single_joint",
                "gripper_left_inner_single_joint",
                "gripper_left_fingertip_1_joint",
                "gripper_left_fingertip_2_joint",
                "gripper_left_fingertip_3_joint",
                "gripper_right_joint",
                "gripper_right_inner_double_joint",
                "gripper_right_motor_single_joint",
                "gripper_right_inner_single_joint",
                "gripper_right_fingertip_1_joint",
                "gripper_right_fingertip_2_joint",
                "gripper_right_fingertip_3_joint"]

            motors_to_remove += [
                "gripper_left_joint_torque",
                "gripper_right_joint_torque"]

            equ_constr_to_remove += ["eq1", "eq2", "eq3", "eq4", "eq5", "eq6"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the XML file of the environment.

        Returns:
            str: Path to the default XML file.

        """
        return cls.get_model_path("talos", "talos.xml")

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: List of observations.

        """

        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_reference", xml_name="reference"),
            ObservationType.JointPos("q_torso_1_joint", xml_name="torso_1_joint"),
            ObservationType.JointPos("q_torso_2_joint", xml_name="torso_2_joint"),
            ObservationType.JointPos("q_head_1_joint", xml_name="head_1_joint"),
            ObservationType.JointPos("q_head_2_joint", xml_name="head_2_joint"),
            ObservationType.JointPos("q_arm_left_1_joint", xml_name="arm_left_1_joint"),
            ObservationType.JointPos("q_arm_left_2_joint", xml_name="arm_left_2_joint"),
            ObservationType.JointPos("q_arm_left_3_joint", xml_name="arm_left_3_joint"),
            ObservationType.JointPos("q_arm_left_4_joint", xml_name="arm_left_4_joint"),
            ObservationType.JointPos("q_arm_left_5_joint", xml_name="arm_left_5_joint"),
            ObservationType.JointPos("q_arm_left_6_joint", xml_name="arm_left_6_joint"),
            ObservationType.JointPos("q_arm_left_7_joint", xml_name="arm_left_7_joint"),
            ObservationType.JointPos("q_gripper_left_joint", xml_name="gripper_left_joint"),
            ObservationType.JointPos("q_gripper_left_inner_double_joint",
                                     xml_name="gripper_left_inner_double_joint"),
            ObservationType.JointPos("q_gripper_left_motor_single_joint",
                                     xml_name="gripper_left_motor_single_joint"),
            ObservationType.JointPos("q_gripper_left_inner_single_joint",
                                     xml_name="gripper_left_inner_single_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_1_joint", xml_name="gripper_left_fingertip_1_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_2_joint", xml_name="gripper_left_fingertip_2_joint"),
            ObservationType.JointPos("q_gripper_left_fingertip_3_joint", xml_name="gripper_left_fingertip_3_joint"),
            ObservationType.JointPos("q_arm_right_1_joint", xml_name="arm_right_1_joint"),
            ObservationType.JointPos("q_arm_right_2_joint", xml_name="arm_right_2_joint"),
            ObservationType.JointPos("q_arm_right_3_joint", xml_name="arm_right_3_joint"),
            ObservationType.JointPos("q_arm_right_4_joint", xml_name="arm_right_4_joint"),
            ObservationType.JointPos("q_arm_right_5_joint", xml_name="arm_right_5_joint"),
            ObservationType.JointPos("q_arm_right_6_joint", xml_name="arm_right_6_joint"),
            ObservationType.JointPos("q_arm_right_7_joint", xml_name="arm_right_7_joint"),
            ObservationType.JointPos("q_gripper_right_joint", xml_name="gripper_right_joint"),
            ObservationType.JointPos("q_gripper_right_inner_double_joint",
                                     xml_name="gripper_right_inner_double_joint"),
            ObservationType.JointPos("q_gripper_right_motor_single_joint",
                                     xml_name="gripper_right_motor_single_joint"),
            ObservationType.JointPos("q_gripper_right_inner_single_joint",
                                     xml_name="gripper_right_inner_single_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_1_joint",
                                     xml_name="gripper_right_fingertip_1_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_2_joint",
                                     xml_name="gripper_right_fingertip_2_joint"),
            ObservationType.JointPos("q_gripper_right_fingertip_3_joint",
                                     xml_name="gripper_right_fingertip_3_joint"),
            ObservationType.JointPos("q_leg_left_1_joint", xml_name="leg_left_1_joint"),
            ObservationType.JointPos("q_leg_left_2_joint", xml_name="leg_left_2_joint"),
            ObservationType.JointPos("q_leg_left_3_joint", xml_name="leg_left_3_joint"),
            ObservationType.JointPos("q_leg_left_4_joint", xml_name="leg_left_4_joint"),
            ObservationType.JointPos("q_leg_left_5_joint", xml_name="leg_left_5_joint"),
            ObservationType.JointPos("q_leg_left_6_joint", xml_name="leg_left_6_joint"),
            ObservationType.JointPos("q_leg_right_1_joint", xml_name="leg_right_1_joint"),
            ObservationType.JointPos("q_leg_right_2_joint", xml_name="leg_right_2_joint"),
            ObservationType.JointPos("q_leg_right_3_joint", xml_name="leg_right_3_joint"),
            ObservationType.JointPos("q_leg_right_4_joint", xml_name="leg_right_4_joint"),
            ObservationType.JointPos("q_leg_right_5_joint", xml_name="leg_right_5_joint"),
            ObservationType.JointPos("q_leg_right_6_joint", xml_name="leg_right_6_joint"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_reference", xml_name="reference"),
            ObservationType.JointVel("dq_torso_1_joint", xml_name="torso_1_joint"),
            ObservationType.JointVel("dq_torso_2_joint", xml_name="torso_2_joint"),
            ObservationType.JointVel("dq_head_1_joint", xml_name="head_1_joint"),
            ObservationType.JointVel("dq_head_2_joint", xml_name="head_2_joint"),
            ObservationType.JointVel("dq_arm_left_1_joint", xml_name="arm_left_1_joint"),
            ObservationType.JointVel("dq_arm_left_2_joint", xml_name="arm_left_2_joint"),
            ObservationType.JointVel("dq_arm_left_3_joint", xml_name="arm_left_3_joint"),
            ObservationType.JointVel("dq_arm_left_4_joint", xml_name="arm_left_4_joint"),
            ObservationType.JointVel("dq_arm_left_5_joint", xml_name="arm_left_5_joint"),
            ObservationType.JointVel("dq_arm_left_6_joint", xml_name="arm_left_6_joint"),
            ObservationType.JointVel("dq_arm_left_7_joint", xml_name="arm_left_7_joint"),
            ObservationType.JointVel("dq_gripper_left_joint", xml_name="gripper_left_joint"),
            ObservationType.JointVel("dq_gripper_left_inner_double_joint",
                                     xml_name="gripper_left_inner_double_joint"),
            ObservationType.JointVel("dq_gripper_left_motor_single_joint",
                                     xml_name="gripper_left_motor_single_joint"),
            ObservationType.JointVel("dq_gripper_left_inner_single_joint",
                                     xml_name="gripper_left_inner_single_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_1_joint",
                                     xml_name="gripper_left_fingertip_1_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_2_joint",
                                     xml_name="gripper_left_fingertip_2_joint"),
            ObservationType.JointVel("dq_gripper_left_fingertip_3_joint",
                                     xml_name="gripper_left_fingertip_3_joint"),
            ObservationType.JointVel("dq_arm_right_1_joint", xml_name="arm_right_1_joint"),
            ObservationType.JointVel("dq_arm_right_2_joint", xml_name="arm_right_2_joint"),
            ObservationType.JointVel("dq_arm_right_3_joint", xml_name="arm_right_3_joint"),
            ObservationType.JointVel("dq_arm_right_4_joint", xml_name="arm_right_4_joint"),
            ObservationType.JointVel("dq_arm_right_5_joint", xml_name="arm_right_5_joint"),
            ObservationType.JointVel("dq_arm_right_6_joint", xml_name="arm_right_6_joint"),
            ObservationType.JointVel("dq_arm_right_7_joint", xml_name="arm_right_7_joint"),
            ObservationType.JointVel("dq_gripper_right_joint", xml_name="gripper_right_joint"),
            ObservationType.JointVel("dq_gripper_right_inner_double_joint",
                                     xml_name="gripper_right_inner_double_joint"),
            ObservationType.JointVel("dq_gripper_right_motor_single_joint",
                                     xml_name="gripper_right_motor_single_joint"),
            ObservationType.JointVel("dq_gripper_right_inner_single_joint",
                                     xml_name="gripper_right_inner_single_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_1_joint",
                                     xml_name="gripper_right_fingertip_1_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_2_joint",
                                     xml_name="gripper_right_fingertip_2_joint"),
            ObservationType.JointVel("dq_gripper_right_fingertip_3_joint",
                                     xml_name="gripper_right_fingertip_3_joint"),
            ObservationType.JointVel("dq_leg_left_1_joint", xml_name="leg_left_1_joint"),
            ObservationType.JointVel("dq_leg_left_2_joint", xml_name="leg_left_2_joint"),
            ObservationType.JointVel("dq_leg_left_3_joint", xml_name="leg_left_3_joint"),
            ObservationType.JointVel("dq_leg_left_4_joint", xml_name="leg_left_4_joint"),
            ObservationType.JointVel("dq_leg_left_5_joint", xml_name="leg_left_5_joint"),
            ObservationType.JointVel("dq_leg_left_6_joint", xml_name="leg_left_6_joint"),
            ObservationType.JointVel("dq_leg_right_1_joint", xml_name="leg_right_1_joint"),
            ObservationType.JointVel("dq_leg_right_2_joint", xml_name="leg_right_2_joint"),
            ObservationType.JointVel("dq_leg_right_3_joint", xml_name="leg_right_3_joint"),
            ObservationType.JointVel("dq_leg_right_4_joint", xml_name="leg_right_4_joint"),
            ObservationType.JointVel("dq_leg_right_5_joint", xml_name="leg_right_5_joint"),
            ObservationType.JointVel("dq_leg_right_6_joint", xml_name="leg_right_6_joint")
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of actuator names.

        """

        action_spec = [
            "torso_1_joint_torque",
            "torso_2_joint_torque",
            "head_1_joint_torque",
            "head_2_joint_torque",
            "arm_left_1_joint_torque",
            "arm_left_2_joint_torque",
            "arm_left_3_joint_torque",
            "arm_left_4_joint_torque",
            "arm_left_5_joint_torque",
            "arm_left_6_joint_torque",
            "arm_left_7_joint_torque",
            "gripper_left_joint_torque",
            "arm_right_1_joint_torque",
            "arm_right_2_joint_torque",
            "arm_right_3_joint_torque",
            "arm_right_4_joint_torque",
            "arm_right_5_joint_torque",
            "arm_right_6_joint_torque",
            "arm_right_7_joint_torque",
            "gripper_right_joint_torque",
            "leg_left_1_joint_torque",
            "leg_left_2_joint_torque",
            "leg_left_3_joint_torque",
            "leg_left_4_joint_torque",
            "leg_left_5_joint_torque",
            "leg_left_6_joint_torque",
            "leg_right_1_joint_torque",
            "leg_right_2_joint_torque",
            "leg_right_3_joint_torque",
            "leg_right_4_joint_torque",
            "leg_right_5_joint_torque",
            "leg_right_6_joint_torque"
        ]

        return action_spec

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.

        Returns:
            str: Name of the upper body.

        """
        return "torso_2_link"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco XML file.

        Returns:
            str: Name of the root body.

        """
        return "base_link"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint in the Mujoco XML file.

        Returns:
            str: Name of the free joint.

        """

        return "reference"

    @info_property
    def init_qpos(self) -> np.ndarray:
        """
        Returns the initial joint positions (``qpos``) of the robot.

        Returns:
            np.ndarray: Initial ``qpos`` of shape ``(49,)``, i.e., of the full model including the grippers:
            position and orientation quaternion (w, x, y, z) of the floating base followed by the joint
            positions.

        """
        return np.array([0.0, 0.0, 1.08, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @info_property
    def init_qvel(self) -> np.ndarray:
        """
        Returns the initial joint velocities (``qvel``) of the robot.

        Returns:
            np.ndarray: Initial ``qvel`` (all zeros) of shape ``(49,)``.

        """
        return np.zeros(49)

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters.

        """
        return (0.8, 1.3)
