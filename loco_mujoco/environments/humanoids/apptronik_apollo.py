"""
Environment of the Apptronik Apollo humanoid robot (CPU MuJoCo backend).
"""

from typing import Union, List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class Apollo(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Apollo robot. Apollo is a humanoid robot developed by Apptronik.

    The robot has a floating base and 32 actuated joints: 3 neck joints, 3 torso joints, 6 joints per leg and
    7 joints per arm. By default, the observation consists of the pose of the floating base without its x-y
    position (height and orientation quaternion), all joint positions, the linear and angular velocity of the
    floating base, and all joint velocities. The action contains one target per actuated joint, which is
    tracked by a PD controller, unless another ``control_type`` is passed.


    Default Observation Space
    -------------------------

    ============ ================ ================ ==================================== ============================== ===
    Index in Obs Name             ObservationType  Min                                  Max                            Dim
    ============ ================ ================ ==================================== ============================== ===
    0 - 4        q_floating_base  FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_neck_yaw       JointPos         [-1.65806]                           [1.65806]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_neck_roll      JointPos         [-0.785398]                          [0.785398]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_neck_pitch     JointPos         [-0.261799]                          [0.523599]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_torso_pitch    JointPos         [-0.305433]                          [1.35263]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_torso_roll     JointPos         [-0.20944]                           [0.20944]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_torso_yaw      JointPos         [-0.829031]                          [0.829031]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_l_hip_ie       JointPos         [-0.567232]                          [1.09083]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_l_hip_aa       JointPos         [-0.218166]                          [0.741765]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_l_hip_fe       JointPos         [-1.85005]                           [0.476475]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_l_knee_fe      JointPos         [0.0]                                [2.61799]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_l_ankle_ie     JointPos         [-0.654498]                          [0.305433]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_l_ankle_pd     JointPos         [-1.5708]                            [0.436332]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_r_hip_ie       JointPos         [-1.09083]                           [0.567232]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_r_hip_aa       JointPos         [-0.741765]                          [0.218166]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_r_hip_fe       JointPos         [-1.85005]                           [0.476475]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_r_knee_fe      JointPos         [0.0]                                [2.61799]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_r_ankle_ie     JointPos         [-0.305433]                          [0.654498]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_r_ankle_pd     JointPos         [-1.5708]                            [0.436332]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_l_shoulder_aa  JointPos         [-0.122173]                          [1.6057]                       1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_l_shoulder_ie  JointPos         [-0.471239]                          [0.471239]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_l_shoulder_fe  JointPos         [-2.18166]                           [0.610865]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_l_elbow_fe     JointPos         [-2.61799]                           [0.174533]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_l_wrist_roll   JointPos         [-1.65806]                           [1.65806]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_l_wrist_yaw    JointPos         [-0.698]                             [0.698]                        1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_l_wrist_pitch  JointPos         [-0.75]                              [1.588]                        1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_r_shoulder_aa  JointPos         [-1.6057]                            [0.122173]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_r_shoulder_ie  JointPos         [-0.471239]                          [0.471239]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_r_shoulder_fe  JointPos         [-2.18166]                           [0.610865]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_r_elbow_fe     JointPos         [-2.61799]                           [0.174533]                     1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    34           q_r_wrist_roll   JointPos         [-1.65806]                           [1.65806]                      1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    35           q_r_wrist_yaw    JointPos         [-0.698]                             [0.698]                        1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    36           q_r_wrist_pitch  JointPos         [-1.588]                             [0.75]                         1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    37 - 42      dq_floating_base FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_neck_yaw      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_neck_roll     JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_neck_pitch    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_torso_pitch   JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_torso_roll    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_torso_yaw     JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_l_hip_ie      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_l_hip_aa      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_l_hip_fe      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_l_knee_fe     JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_l_ankle_ie    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_l_ankle_pd    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_r_hip_ie      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_r_hip_aa      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_r_hip_fe      JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_r_knee_fe     JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_r_ankle_ie    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_r_ankle_pd    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    61           dq_l_shoulder_aa JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    62           dq_l_shoulder_ie JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    63           dq_l_shoulder_fe JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    64           dq_l_elbow_fe    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    65           dq_l_wrist_roll  JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    66           dq_l_wrist_yaw   JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    67           dq_l_wrist_pitch JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    68           dq_r_shoulder_aa JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    69           dq_r_shoulder_ie JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    70           dq_r_shoulder_fe JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    71           dq_r_elbow_fe    JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    72           dq_r_wrist_roll  JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    73           dq_r_wrist_yaw   JointVel         [-inf]                               [inf]                          1
    ------------ ---------------- ---------------- ------------------------------------ ------------------------------ ---
    74           dq_r_wrist_pitch JointVel         [-inf]                               [inf]                          1
    ============ ================ ================ ==================================== ============================== ===


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
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Model to simulate, either a path to a MuJoCo XML file or an ``MjSpec``
                object. If None, the default model of this environment is used.
            observation_spec (List[Observation]): Observations that make up the observation vector, given as
                observation objects or as dictionaries with a ``type`` key naming the observation type and its
                parameters as the remaining keys. If None, the default observation space is used.
            actuation_spec (List[str]): Names of the actuators in the model that are controlled by the action.
                If None, the default action space is used.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.base.LocoEnv`. If no
                ``control_type`` is given, ``PDControl`` with the gains :attr:`p_gains` and :attr:`d_gains` is
                used. The extra key ``mjx_reduced_contacts`` (bool, default True) only affects the Mjx variant:
                if True, the contacts are reduced by ``_modify_spec_for_mjx``; if False, the collision
                geometries of the model are kept.

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

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=self.p_gains, d_gain=self.d_gains, scale_action_to_jnt_limits=False)

        # modify the specification if needed
        # mjx_reduced_contacts=False keeps the robot's own collision geoms on Mjx/MjWarp instead of the
        # simplified feet-only contacts of _modify_spec_for_mjx
        if kwargs.pop("mjx_reduced_contacts", True) and self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

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
            ObservationType.FreeJointPosNoXY("q_floating_base", xml_name="floating_base"),
            ObservationType.JointPos("q_neck_yaw", xml_name="neck_yaw"),
            ObservationType.JointPos("q_neck_roll", xml_name="neck_roll"),
            ObservationType.JointPos("q_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointPos("q_torso_pitch", xml_name="torso_pitch"),
            ObservationType.JointPos("q_torso_roll", xml_name="torso_roll"),
            ObservationType.JointPos("q_torso_yaw", xml_name="torso_yaw"),
            ObservationType.JointPos("q_l_hip_ie", xml_name="l_hip_ie"),
            ObservationType.JointPos("q_l_hip_aa", xml_name="l_hip_aa"),
            ObservationType.JointPos("q_l_hip_fe", xml_name="l_hip_fe"),
            ObservationType.JointPos("q_l_knee_fe", xml_name="l_knee_fe"),
            ObservationType.JointPos("q_l_ankle_ie", xml_name="l_ankle_ie"),
            ObservationType.JointPos("q_l_ankle_pd", xml_name="l_ankle_pd"),
            ObservationType.JointPos("q_r_hip_ie", xml_name="r_hip_ie"),
            ObservationType.JointPos("q_r_hip_aa", xml_name="r_hip_aa"),
            ObservationType.JointPos("q_r_hip_fe", xml_name="r_hip_fe"),
            ObservationType.JointPos("q_r_knee_fe", xml_name="r_knee_fe"),
            ObservationType.JointPos("q_r_ankle_ie", xml_name="r_ankle_ie"),
            ObservationType.JointPos("q_r_ankle_pd", xml_name="r_ankle_pd"),
            ObservationType.JointPos("q_l_shoulder_aa", xml_name="l_shoulder_aa"),
            ObservationType.JointPos("q_l_shoulder_ie", xml_name="l_shoulder_ie"),
            ObservationType.JointPos("q_l_shoulder_fe", xml_name="l_shoulder_fe"),
            ObservationType.JointPos("q_l_elbow_fe", xml_name="l_elbow_fe"),
            ObservationType.JointPos("q_l_wrist_roll", xml_name="l_wrist_roll"),
            ObservationType.JointPos("q_l_wrist_yaw", xml_name="l_wrist_yaw"),
            ObservationType.JointPos("q_l_wrist_pitch", xml_name="l_wrist_pitch"),
            ObservationType.JointPos("q_r_shoulder_aa", xml_name="r_shoulder_aa"),
            ObservationType.JointPos("q_r_shoulder_ie", xml_name="r_shoulder_ie"),
            ObservationType.JointPos("q_r_shoulder_fe", xml_name="r_shoulder_fe"),
            ObservationType.JointPos("q_r_elbow_fe", xml_name="r_elbow_fe"),
            ObservationType.JointPos("q_r_wrist_roll", xml_name="r_wrist_roll"),
            ObservationType.JointPos("q_r_wrist_yaw", xml_name="r_wrist_yaw"),
            ObservationType.JointPos("q_r_wrist_pitch", xml_name="r_wrist_pitch"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_floating_base", xml_name="floating_base"),
            ObservationType.JointVel("dq_neck_yaw", xml_name="neck_yaw"),
            ObservationType.JointVel("dq_neck_roll", xml_name="neck_roll"),
            ObservationType.JointVel("dq_neck_pitch", xml_name="neck_pitch"),
            ObservationType.JointVel("dq_torso_pitch", xml_name="torso_pitch"),
            ObservationType.JointVel("dq_torso_roll", xml_name="torso_roll"),
            ObservationType.JointVel("dq_torso_yaw", xml_name="torso_yaw"),
            ObservationType.JointVel("dq_l_hip_ie", xml_name="l_hip_ie"),
            ObservationType.JointVel("dq_l_hip_aa", xml_name="l_hip_aa"),
            ObservationType.JointVel("dq_l_hip_fe", xml_name="l_hip_fe"),
            ObservationType.JointVel("dq_l_knee_fe", xml_name="l_knee_fe"),
            ObservationType.JointVel("dq_l_ankle_ie", xml_name="l_ankle_ie"),
            ObservationType.JointVel("dq_l_ankle_pd", xml_name="l_ankle_pd"),
            ObservationType.JointVel("dq_r_hip_ie", xml_name="r_hip_ie"),
            ObservationType.JointVel("dq_r_hip_aa", xml_name="r_hip_aa"),
            ObservationType.JointVel("dq_r_hip_fe", xml_name="r_hip_fe"),
            ObservationType.JointVel("dq_r_knee_fe", xml_name="r_knee_fe"),
            ObservationType.JointVel("dq_r_ankle_ie", xml_name="r_ankle_ie"),
            ObservationType.JointVel("dq_r_ankle_pd", xml_name="r_ankle_pd"),
            ObservationType.JointVel("dq_l_shoulder_aa", xml_name="l_shoulder_aa"),
            ObservationType.JointVel("dq_l_shoulder_ie", xml_name="l_shoulder_ie"),
            ObservationType.JointVel("dq_l_shoulder_fe", xml_name="l_shoulder_fe"),
            ObservationType.JointVel("dq_l_elbow_fe", xml_name="l_elbow_fe"),
            ObservationType.JointVel("dq_l_wrist_roll", xml_name="l_wrist_roll"),
            ObservationType.JointVel("dq_l_wrist_yaw", xml_name="l_wrist_yaw"),
            ObservationType.JointVel("dq_l_wrist_pitch", xml_name="l_wrist_pitch"),
            ObservationType.JointVel("dq_r_shoulder_aa", xml_name="r_shoulder_aa"),
            ObservationType.JointVel("dq_r_shoulder_ie", xml_name="r_shoulder_ie"),
            ObservationType.JointVel("dq_r_shoulder_fe", xml_name="r_shoulder_fe"),
            ObservationType.JointVel("dq_r_elbow_fe", xml_name="r_elbow_fe"),
            ObservationType.JointVel("dq_r_wrist_roll", xml_name="r_wrist_roll"),
            ObservationType.JointVel("dq_r_wrist_yaw", xml_name="r_wrist_yaw"),
            ObservationType.JointVel("dq_r_wrist_pitch", xml_name="r_wrist_pitch"),
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

        action_spec = ["neck_yaw", "neck_roll", "neck_pitch", "torso_pitch", "torso_roll", "torso_yaw", "l_hip_ie",
                       "l_hip_aa", "l_hip_fe", "l_knee_fe", "l_ankle_ie", "l_ankle_pd", "r_hip_ie", "r_hip_aa",
                       "r_hip_fe", "r_knee_fe", "r_ankle_ie", "r_ankle_pd", "l_shoulder_aa", "l_shoulder_ie",
                       "l_shoulder_fe", "l_elbow_fe", "l_wrist_roll", "l_wrist_yaw", "l_wrist_pitch", "r_shoulder_aa",
                       "r_shoulder_ie", "r_shoulder_fe", "r_elbow_fe", "r_wrist_roll", "r_wrist_yaw", "r_wrist_pitch"
                       ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        Returns:
            str: Path to the default xml file.

        """
        return cls.get_model_path("apptronik_apollo", "apptronik_apollo.xml")

    @info_property
    def p_gains(self) -> Union[float, List[float]]:
        """
        Returns the proportional gains for the default PD controller.

        Returns:
            Union[float, List[float]]: One proportional gain per actuator, in the order of the default action
            specification.

        """
        return [28, 9, 8, 1525, 2052, 600, 595, 1880, 1047, 606, 420, 882, 595, 1880, 1047, 606, 420, 882, 395, 530,
                277, 312, 47, 20, 18, 395, 530, 277, 312, 47, 20, 18]

    @info_property
    def d_gains(self) -> Union[float, List[float]]:
        """
        Returns the derivative gain used for the default PD controller.

        Returns:
            Union[float, List[float]]: Derivative gain, here a single value used for all joints.

        """
        return 0.0

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml.

        Returns:
            str: Name of the upper body.

        """
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root set in the Mujoco xml.

        Returns:
            str: Name of the free joint.

        """
        return "floating_base"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters.

        """
        return (0.6, 1.5)
