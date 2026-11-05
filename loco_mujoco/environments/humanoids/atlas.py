from typing import Union, List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core.utils import info_property
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid


class Atlas(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Atlas robot.


    Default Observation Space
    -------------------------

    ============ ================== ================ ==================================== ============================== ===
    Index in Obs Name               ObservationType  Min                                  Max                            Dim
    ============ ================== ================ ==================================== ============================== ===
    0 - 4        q_root             FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    5            q_back_bkz         JointPos         [-0.663225]                          [0.663225]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    6            q_back_bkx         JointPos         [-0.523599]                          [0.523599]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    7            q_back_bky         JointPos         [-0.219388]                          [0.538783]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    8            q_l_arm_shz        JointPos         [-1.5708]                            [0.785398]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    9            q_l_arm_shx        JointPos         [-1.5708]                            [1.5708]                       1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    10           q_l_arm_ely        JointPos         [0.0]                                [3.14159]                      1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    11           q_l_arm_elx        JointPos         [0.0]                                [2.35619]                      1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    12           q_l_arm_wry        JointPos         [-3.011]                             [3.011]                        1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    13           q_l_arm_wrx        JointPos         [-1.7628]                            [1.7628]                       1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    14           q_r_arm_shz        JointPos         [-0.785398]                          [1.5708]                       1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    15           q_r_arm_shx        JointPos         [-1.5708]                            [1.5708]                       1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    16           q_r_arm_ely        JointPos         [0.0]                                [3.14159]                      1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    17           q_r_arm_elx        JointPos         [-2.35619]                           [0.0]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    18           q_r_arm_wry        JointPos         [-3.011]                             [3.011]                        1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    19           q_r_arm_wrx        JointPos         [-1.7628]                            [1.7628]                       1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    20           q_hip_flexion_r    JointPos         [-0.786794]                          [0.786794]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    21           q_hip_adduction_r  JointPos         [-0.523599]                          [0.523599]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    22           q_hip_rotation_r   JointPos         [-1.61234]                           [1.61234]                      1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    23           q_knee_angle_r     JointPos         [-2.35637]                           [0.174]                        1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    24           q_ankle_angle_r    JointPos         [-1.0]                               [1.0]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    25           q_r_leg_akx        JointPos         [-0.8]                               [0.8]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    26           q_hip_flexion_l    JointPos         [-0.786794]                          [0.786794]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    27           q_hip_adduction_l  JointPos         [-0.523599]                          [0.523599]                     1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    28           q_hip_rotation_l   JointPos         [-1.61234]                           [1.61234]                      1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    29           q_knee_angle_l     JointPos         [-2.35637]                           [0.174]                        1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    30           q_ankle_angle_l    JointPos         [-1.0]                               [1.0]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    31           q_l_leg_akx        JointPos         [-0.8]                               [0.8]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    32 - 37      dq_root            FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    38           dq_back_bkz        JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    39           dq_back_bkx        JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    40           dq_back_bky        JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    41           dq_l_arm_shz       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    42           dq_l_arm_shx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    43           dq_l_arm_ely       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    44           dq_l_arm_elx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    45           dq_l_arm_wry       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    46           dq_l_arm_wrx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    47           dq_r_arm_shz       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    48           dq_r_arm_shx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    49           dq_r_arm_ely       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    50           dq_r_arm_elx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    51           dq_r_arm_wry       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    52           dq_r_arm_wrx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    53           dq_hip_flexion_r   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    54           dq_hip_adduction_r JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    55           dq_hip_rotation_r  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    56           dq_knee_angle_r    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    57           dq_ankle_angle_r   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    58           dq_r_leg_akx       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    59           dq_hip_flexion_l   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    60           dq_hip_adduction_l JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    61           dq_hip_rotation_l  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    62           dq_knee_angle_l    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    63           dq_ankle_angle_l   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    64           dq_l_leg_akx       JointVel         [-inf]                               [inf]                          1
    ============ ================== ================ ==================================== ============================== ===

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
    =============== ==== ===


    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_arms: bool = False, disable_back_joint: bool = False, spec: Union[str, MjSpec] = None,
                 observation_spec: List[ObservationType] = None, actuation_spec: List[str] = None, **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_arms (bool): If True, all arm joints are removed and the respective actuators are removed from the action specification.
            disable_back_joint (bool): If True, the back joint is removed and the respective actuators are removed from the action specification.
            spec (Union[str, MjSpec]): Specification of the environment. It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            observation_spec (List[ObservationType]): Observation specification.
            actuation_spec (List[str]): Action specification.
            **kwargs: Additional arguments.
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

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=self.p_gains, d_gain=self.d_gains, scale_action_to_jnt_limits=False)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:

            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove, motors_to_remove, equ_constr_to_remove)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Function that specifies which joints, motors, and equality constraints should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove, and names of equality constraints to remove.
        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["l_arm_shz", "l_arm_shx", "l_arm_ely", "l_arm_elx", "l_arm_wry", "l_arm_wrx",
                                 "r_arm_shz", "r_arm_shx", "r_arm_ely", "r_arm_elx", "r_arm_wry", "r_arm_wrx"]
            motors_to_remove += ["l_arm_shz_actuator", "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator",
                                 "l_arm_wry_actuator", "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                                 "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator"]

        if self._disable_back_joint:
            joints_to_remove += ["back_bkz", "back_bky", "back_bkx"]
            motors_to_remove += ["back_bkz_actuator", "back_bky_actuator", "back_bkx_actuator"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """
        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_back_bkz", xml_name="back_bkz"),
                            ObservationType.JointPos("q_back_bkx", xml_name="back_bkx"),
                            ObservationType.JointPos("q_back_bky", xml_name="back_bky"),
                            ObservationType.JointPos("q_l_arm_shz", xml_name="l_arm_shz"),
                            ObservationType.JointPos("q_l_arm_shx", xml_name="l_arm_shx"),
                            ObservationType.JointPos("q_l_arm_ely", xml_name="l_arm_ely"),
                            ObservationType.JointPos("q_l_arm_elx", xml_name="l_arm_elx"),
                            ObservationType.JointPos("q_l_arm_wry", xml_name="l_arm_wry"),
                            ObservationType.JointPos("q_l_arm_wrx", xml_name="l_arm_wrx"),
                            ObservationType.JointPos("q_r_arm_shz", xml_name="r_arm_shz"),
                            ObservationType.JointPos("q_r_arm_shx", xml_name="r_arm_shx"),
                            ObservationType.JointPos("q_r_arm_ely", xml_name="r_arm_ely"),
                            ObservationType.JointPos("q_r_arm_elx", xml_name="r_arm_elx"),
                            ObservationType.JointPos("q_r_arm_wry", xml_name="r_arm_wry"),
                            ObservationType.JointPos("q_r_arm_wrx", xml_name="r_arm_wrx"),
                            ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointPos("q_r_leg_akx", xml_name="r_leg_akx"),
                            ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),
                            ObservationType.JointPos("q_l_leg_akx", xml_name="l_leg_akx"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_back_bkz", xml_name="back_bkz"),
                            ObservationType.JointVel("dq_back_bkx", xml_name="back_bkx"),
                            ObservationType.JointVel("dq_back_bky", xml_name="back_bky"),
                            ObservationType.JointVel("dq_l_arm_shz", xml_name="l_arm_shz"),
                            ObservationType.JointVel("dq_l_arm_shx", xml_name="l_arm_shx"),
                            ObservationType.JointVel("dq_l_arm_ely", xml_name="l_arm_ely"),
                            ObservationType.JointVel("dq_l_arm_elx", xml_name="l_arm_elx"),
                            ObservationType.JointVel("dq_l_arm_wry", xml_name="l_arm_wry"),
                            ObservationType.JointVel("dq_l_arm_wrx", xml_name="l_arm_wrx"),
                            ObservationType.JointVel("dq_r_arm_shz", xml_name="r_arm_shz"),
                            ObservationType.JointVel("dq_r_arm_shx", xml_name="r_arm_shx"),
                            ObservationType.JointVel("dq_r_arm_ely", xml_name="r_arm_ely"),
                            ObservationType.JointVel("dq_r_arm_elx", xml_name="r_arm_elx"),
                            ObservationType.JointVel("dq_r_arm_wry", xml_name="r_arm_wry"),
                            ObservationType.JointVel("dq_r_arm_wrx", xml_name="r_arm_wrx"),
                            ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointVel("dq_r_leg_akx", xml_name="r_leg_akx"),
                            ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l"),
                            ObservationType.JointVel("dq_l_leg_akx", xml_name="l_leg_akx")]

        return observation_spec

    @info_property
    def p_gains(self) -> Union[float, List[float]]:
        """
        Returns the proportional gains for the default PD controller.
        """
        return 100.0

    @info_property
    def d_gains(self) -> Union[float, List[float]]:
        """
        Returns the derivative gains for the default PD controller.
        """
        return 1.0

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of actuator names.
        """

        action_spec = ["back_bkz_actuator", "back_bky_actuator", "back_bkx_actuator", "l_arm_shz_actuator",
                       "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator", "l_arm_wry_actuator",
                       "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                       "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator",
                       "hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator",
                       "knee_angle_r_actuator", "ankle_angle_r_actuator", "r_leg_akx_actuator", "hip_flexion_l_actuator",
                       "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator",
                       "ankle_angle_l_actuator", "l_leg_akx_actuator"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.
        """
        return cls.get_model_path("atlas", "atlas.xml")

    @info_property
    def upper_body_xml_name(self) -> str:
        return "utorso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.0, 1.0)