from typing import Union, List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class BoosterT1(BaseRobotHumanoid):

    """

    Description
    ------------
    Environment of the Booster T1 robot. The Booster T1 is a humanoid robot from Booster Robotics.


    Default Observation Space
    -------------------------

    ============ ======================= ================ ==================================== ============================== ===
    Index in Obs Name                    ObservationType  Min                                  Max                            Dim
    ============ ======================= ================ ==================================== ============================== ===
    0 - 4        q_root                  FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_AAHead_yaw            JointPos         [-1.57]                              [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_Head_pitch            JointPos         [-0.35]                              [1.22]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_Left_Shoulder_Pitch   JointPos         [-3.31]                              [1.22]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_Left_Shoulder_Roll    JointPos         [-1.74]                              [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_Left_Elbow_Pitch      JointPos         [-2.27]                              [2.27]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_Left_Elbow_Yaw        JointPos         [-2.44]                              [0.0]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_Right_Shoulder_Pitch  JointPos         [-3.31]                              [1.22]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_Right_Shoulder_Roll   JointPos         [-1.57]                              [1.74]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_Right_Elbow_Pitch     JointPos         [-2.27]                              [2.27]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_Right_Elbow_Yaw       JointPos         [0.0]                                [2.44]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_Waist                 JointPos         [-1.57]                              [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_Left_Hip_Pitch        JointPos         [-1.8]                               [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_Left_Hip_Roll         JointPos         [-0.2]                               [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_Left_Hip_Yaw          JointPos         [-1.0]                               [1.0]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_Left_Knee_Pitch       JointPos         [0.0]                                [2.34]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_Left_Ankle_Pitch      JointPos         [-0.87]                              [0.35]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_Left_Ankle_Roll       JointPos         [-0.44]                              [0.44]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_Right_Hip_Pitch       JointPos         [-1.8]                               [1.57]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_Right_Hip_Roll        JointPos         [-1.57]                              [0.2]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_Right_Hip_Yaw         JointPos         [-1.0]                               [1.0]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_Right_Knee_Pitch      JointPos         [0.0]                                [2.34]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_Right_Ankle_Pitch     JointPos         [-0.87]                              [0.35]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_Right_Ankle_Roll      JointPos         [-0.44]                              [0.44]                         1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    28 - 33      dq_root                 FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_AAHead_yaw           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    35           dq_Head_pitch           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           dq_Left_Shoulder_Pitch  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           dq_Left_Shoulder_Roll   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           dq_Left_Elbow_Pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           dq_Left_Elbow_Yaw       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_Right_Shoulder_Pitch JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_Right_Shoulder_Roll  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_Right_Elbow_Pitch    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_Right_Elbow_Yaw      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_Waist                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_Left_Hip_Pitch       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_Left_Hip_Roll        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_Left_Hip_Yaw         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_Left_Knee_Pitch      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_Left_Ankle_Pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_Left_Ankle_Roll      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_Right_Hip_Pitch      JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_Right_Hip_Roll       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_Right_Hip_Yaw        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_Right_Knee_Pitch     JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_Right_Ankle_Pitch    JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_Right_Ankle_Roll     JointVel         [-inf]                               [inf]                          1
    ============ ======================= ================ ==================================== ============================== ===

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
    =============== ==== ===

    Methods
    ---------

    """

    mjx_enabled = False

    def __init__(self, spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None, **kwargs):
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            observation_spec (List[Observation]): Observation specification.
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

        # uses PD control by default
        if "control_type" not in kwargs.keys():
            kwargs["control_type"] = "PDControl"
            kwargs["control_params"] = dict(p_gain=self.p_gains, d_gain=self.d_gains, scale_action_to_jnt_limits=False)

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
            A list of observations.
        """
        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_AAHead_yaw", xml_name="AAHead_yaw"),
            ObservationType.JointPos("q_Head_pitch", xml_name="Head_pitch"),
            ObservationType.JointPos("q_Left_Shoulder_Pitch", xml_name="Left_Shoulder_Pitch"),
            ObservationType.JointPos("q_Left_Shoulder_Roll", xml_name="Left_Shoulder_Roll"),
            ObservationType.JointPos("q_Left_Elbow_Pitch", xml_name="Left_Elbow_Pitch"),
            ObservationType.JointPos("q_Left_Elbow_Yaw", xml_name="Left_Elbow_Yaw"),
            ObservationType.JointPos("q_Right_Shoulder_Pitch", xml_name="Right_Shoulder_Pitch"),
            ObservationType.JointPos("q_Right_Shoulder_Roll", xml_name="Right_Shoulder_Roll"),
            ObservationType.JointPos("q_Right_Elbow_Pitch", xml_name="Right_Elbow_Pitch"),
            ObservationType.JointPos("q_Right_Elbow_Yaw", xml_name="Right_Elbow_Yaw"),
            ObservationType.JointPos("q_Waist", xml_name="Waist"),
            ObservationType.JointPos("q_Left_Hip_Pitch", xml_name="Left_Hip_Pitch"),
            ObservationType.JointPos("q_Left_Hip_Roll", xml_name="Left_Hip_Roll"),
            ObservationType.JointPos("q_Left_Hip_Yaw", xml_name="Left_Hip_Yaw"),
            ObservationType.JointPos("q_Left_Knee_Pitch", xml_name="Left_Knee_Pitch"),
            ObservationType.JointPos("q_Left_Ankle_Pitch", xml_name="Left_Ankle_Pitch"),
            ObservationType.JointPos("q_Left_Ankle_Roll", xml_name="Left_Ankle_Roll"),
            ObservationType.JointPos("q_Right_Hip_Pitch", xml_name="Right_Hip_Pitch"),
            ObservationType.JointPos("q_Right_Hip_Roll", xml_name="Right_Hip_Roll"),
            ObservationType.JointPos("q_Right_Hip_Yaw", xml_name="Right_Hip_Yaw"),
            ObservationType.JointPos("q_Right_Knee_Pitch", xml_name="Right_Knee_Pitch"),
            ObservationType.JointPos("q_Right_Ankle_Pitch", xml_name="Right_Ankle_Pitch"),
            ObservationType.JointPos("q_Right_Ankle_Roll", xml_name="Right_Ankle_Roll"),

            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_AAHead_yaw", xml_name="AAHead_yaw"),
            ObservationType.JointVel("dq_Head_pitch", xml_name="Head_pitch"),
            ObservationType.JointVel("dq_Left_Shoulder_Pitch", xml_name="Left_Shoulder_Pitch"),
            ObservationType.JointVel("dq_Left_Shoulder_Roll", xml_name="Left_Shoulder_Roll"),
            ObservationType.JointVel("dq_Left_Elbow_Pitch", xml_name="Left_Elbow_Pitch"),
            ObservationType.JointVel("dq_Left_Elbow_Yaw", xml_name="Left_Elbow_Yaw"),
            ObservationType.JointVel("dq_Right_Shoulder_Pitch", xml_name="Right_Shoulder_Pitch"),
            ObservationType.JointVel("dq_Right_Shoulder_Roll", xml_name="Right_Shoulder_Roll"),
            ObservationType.JointVel("dq_Right_Elbow_Pitch", xml_name="Right_Elbow_Pitch"),
            ObservationType.JointVel("dq_Right_Elbow_Yaw", xml_name="Right_Elbow_Yaw"),
            ObservationType.JointVel("dq_Waist", xml_name="Waist"),
            ObservationType.JointVel("dq_Left_Hip_Pitch", xml_name="Left_Hip_Pitch"),
            ObservationType.JointVel("dq_Left_Hip_Roll", xml_name="Left_Hip_Roll"),
            ObservationType.JointVel("dq_Left_Hip_Yaw", xml_name="Left_Hip_Yaw"),
            ObservationType.JointVel("dq_Left_Knee_Pitch", xml_name="Left_Knee_Pitch"),
            ObservationType.JointVel("dq_Left_Ankle_Pitch", xml_name="Left_Ankle_Pitch"),
            ObservationType.JointVel("dq_Left_Ankle_Roll", xml_name="Left_Ankle_Roll"),
            ObservationType.JointVel("dq_Right_Hip_Pitch", xml_name="Right_Hip_Pitch"),
            ObservationType.JointVel("dq_Right_Hip_Roll", xml_name="Right_Hip_Roll"),
            ObservationType.JointVel("dq_Right_Hip_Yaw", xml_name="Right_Hip_Yaw"),
            ObservationType.JointVel("dq_Right_Knee_Pitch", xml_name="Right_Knee_Pitch"),
            ObservationType.JointVel("dq_Right_Ankle_Pitch", xml_name="Right_Ankle_Pitch"),
            ObservationType.JointVel("dq_Right_Ankle_Roll", xml_name="Right_Ankle_Roll"),

        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of actuator names.

        """
        action_spec = ["AAHead_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
                       "Left_Elbow_Yaw", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch",
                       "Right_Elbow_Yaw", "Waist", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
                       "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Hip_Pitch", "Right_Hip_Roll",
                       "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        """
        return cls.get_model_path("booster_t1", "booster_t1.xml")

    @info_property
    def p_gains(self) -> Union[float, List[float]]:
        """
        Returns the proportional gains for the default PD controller.

        """
        return 75.0

    @info_property
    def d_gains(self) -> Union[float, List[float]]:
        """
        Returns the derivative gains for the default PD controller.

        """
        return 0.0

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml.

        """
        return self.root_body_name

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml.

        """
        return "Trunk"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root body in the Mujoco xml.

        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        """
        return (0.3, 1.0)
