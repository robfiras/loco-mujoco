"""
Environment of the Booster T1 humanoid robot (CPU/MuJoCo version).
"""

from typing import Union, List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class BoosterT1(BaseRobotHumanoid):

    """
    Environment of the Booster T1 humanoid robot.

    Description
    -----------

    MuJoCo environment of the Booster T1, a humanoid robot by Booster Robotics. This class uses the CPU (MuJoCo)
    backend; :class:`~loco_mujoco.environments.humanoids.boostert1_mjx.MjxBoosterT1` is its Mjx counterpart.

    The model (``booster_t1/booster_t1.xml``) has a floating base and 23 actuated joints: two head joints, four joints
    per arm, the waist joint and six joints per leg.

    By default, the observation has 57 dimensions: the height and the orientation quaternion of the floating base (its
    horizontal position is left out), the 23 joint positions, the 6D velocity of the floating base and the 23 joint
    velocities. The action has 23 dimensions, one per actuated joint. Unless ``control_type`` is given, a PD
    controller (``PDControl``) with the gains in :attr:`p_gains` and :attr:`d_gains` is used: the action is added to
    the nominal joint positions to obtain the position targets of the controller, without scaling to the joint limits.
    Both spaces are listed below and can be changed with the constructor arguments ``observation_spec`` and
    ``actuation_spec``.

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
            spec (Union[str, MjSpec], optional): Model of the robot, either the path to a MuJoCo XML file or an
                ``MjSpec`` object. If None, the default XML file is used (see :meth:`get_default_xml_file_path`).
            observation_spec (List[Observation], optional): Observations that make up the observation space, as
                :class:`~loco_mujoco.core.observations.base.Observation` objects or as dictionaries that are parsed
                with ``parse_observation_spec``. If None, the default observation space listed in the class docstring
                is used.
            actuation_spec (List[str], optional): Names of the actuators in the model that make up the action
                space. If None, the default action space listed in the class docstring is used.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.base.LocoEnv`. The
                keyword ``mjx_reduced_contacts`` (bool, default True) is consumed here: if False, the Mjx version of
                this environment keeps the collision geoms of the model instead of the reduced set of contacts of
                ``_modify_spec_for_mjx``. It has no effect on the CPU version.
                If ``control_type`` is not given, ``PDControl`` is used with the gains of this class.

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
        Returns the default observation specification of the environment.

        The default observation consists of the height and orientation of the floating base, the joint positions,
        the velocity of the floating base and the joint velocities.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: The list of default observations.

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
        Returns the default action specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: The names of the actuators that make up the default action space.

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
        Returns the default path to the XML file of the Booster T1 model.

        Returns:
            str: Path to the XML file.

        """
        return cls.get_model_path("booster_t1", "booster_t1.xml")

    @info_property
    def p_gains(self) -> Union[float, List[float]]:
        """
        Returns the proportional gain of the default PD controller.

        Returns:
            Union[float, List[float]]: The proportional gain, here one value shared by all actuators.

        """
        return 75.0

    @info_property
    def d_gains(self) -> Union[float, List[float]]:
        """
        Returns the derivative gain of the default PD controller.

        Returns:
            Union[float, List[float]]: The derivative gain, here one value shared by all actuators.

        """
        return 0.0

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the XML file. For this robot, it is the root body.

        Returns:
            str: Name of the upper body.

        """
        return self.root_body_name

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the XML file.

        Returns:
            str: Name of the root body.

        """
        return "Trunk"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root in the XML file.

        Returns:
            str: Name of the free joint.

        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the range of root heights in which the robot is considered healthy.

        This is only used when the ``HeightBasedTerminalStateHandler`` is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters.

        """
        return (0.3, 1.0)
