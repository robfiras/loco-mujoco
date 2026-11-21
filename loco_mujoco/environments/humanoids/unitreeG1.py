"""
Environment of the Unitree G1 humanoid robot (CPU/MuJoCo version).
"""

from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec
import numpy as np

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core. utils import info_property


class UnitreeG1(BaseRobotHumanoid):

    """
    Environment of the Unitree G1 humanoid robot.

    Description
    -----------

    MuJoCo environment of the Unitree G1, a humanoid robot by Unitree Robotics. This class uses the CPU (MuJoCo)
    backend; :class:`~loco_mujoco.environments.humanoids.unitreeG1_mjx.MjxUnitreeG1` is its Mjx counterpart.

    The default model (``unitree_g1/g1_23dof.xml``) has a floating base and 23 actuated joints: six joints per leg,
    the waist yaw joint and five joints per arm (three shoulder joints, the elbow and the wrist roll joint).

    By default, the observation has 57 dimensions: the height and the orientation quaternion of the floating base (its
    horizontal position is left out), the 23 joint positions, the 6D velocity of the floating base and the 23 joint
    velocities. The action has 23 dimensions, one per motor, each in ``[-1, 1]``. The default control function
    (``DefaultControl``) rescales it to the control range of the motors. Both spaces are listed below and can be
    changed with the constructor arguments ``observation_spec`` and ``actuation_spec``.

    The options ``disable_arms`` and ``disable_back_joint`` remove the joints and actuators named in
    :meth:`_get_xml_modifications` from the model, together with the corresponding observations.

    Default Observation Space
    -------------------------
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
            disable_arms (bool): If True, the arm joints and actuators named in :meth:`_get_xml_modifications` are
                removed together with their observations, and the arms are re-oriented to prevent collisions with
                the hips.
            disable_back_joint (bool): If True, the back joint and actuator named in :meth:`_get_xml_modifications`
                are removed together with the observations of that joint.
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
        # mjx_reduced_contacts=False keeps the robot's own collision geoms on Mjx/MjWarp instead of the
        # simplified feet-only contacts of _modify_spec_for_mjx
        if kwargs.pop("mjx_reduced_contacts", True) and self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_xml_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators and equality constraints are removed from the MuJoCo specification.

        The lists depend on ``disable_arms`` and ``disable_back_joint``. Names that do not exist in the model are
        ignored when the specification is modified.

        Returns:
            Tuple[List[str], List[str], List[str]]: The names of the joints to remove, of the actuators to remove
            and of the equality constraints to remove (always empty for this robot).

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_pitch_joint", "right_elbow_roll_joint",
                                 "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_pitch_joint", "left_elbow_roll_joint"]
            # actuators are named without the \"_joint\" suffix in the XML
            motors_to_remove += ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                                 "right_elbow_pitch", "right_elbow_roll",
                                 "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                                 "left_elbow_pitch", "left_elbow_roll"]

        if self._disable_back_joint:
            joints_to_remove += ["torso_joint"]
            motors_to_remove += ["torso_joint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

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
                            ObservationType.JointPos("q_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointPos("q_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointPos("q_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointPos("q_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointPos("q_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointPos("q_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointPos("q_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_left_hip_pitch_joint", xml_name="left_hip_pitch_joint"),
                            ObservationType.JointVel("dq_left_hip_roll_joint", xml_name="left_hip_roll_joint"),
                            ObservationType.JointVel("dq_left_hip_yaw_joint", xml_name="left_hip_yaw_joint"),
                            ObservationType.JointVel("dq_left_knee_joint", xml_name="left_knee_joint"),
                            ObservationType.JointVel("dq_left_ankle_pitch_joint", xml_name="left_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_left_ankle_roll_joint", xml_name="left_ankle_roll_joint"),
                            ObservationType.JointVel("dq_right_hip_pitch_joint", xml_name="right_hip_pitch_joint"),
                            ObservationType.JointVel("dq_right_hip_roll_joint", xml_name="right_hip_roll_joint"),
                            ObservationType.JointVel("dq_right_hip_yaw_joint", xml_name="right_hip_yaw_joint"),
                            ObservationType.JointVel("dq_right_knee_joint", xml_name="right_knee_joint"),
                            ObservationType.JointVel("dq_right_ankle_pitch_joint", xml_name="right_ankle_pitch_joint"),
                            ObservationType.JointVel("dq_right_ankle_roll_joint", xml_name="right_ankle_roll_joint"),
                            ObservationType.JointVel("dq_waist_yaw_joint", xml_name="waist_yaw_joint"),
                            ObservationType.JointVel("dq_left_shoulder_pitch_joint", xml_name="left_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_left_shoulder_roll_joint", xml_name="left_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_left_shoulder_yaw_joint", xml_name="left_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_left_elbow_joint", xml_name="left_elbow_joint"),
                            ObservationType.JointVel("dq_left_wrist_roll_joint", xml_name="left_wrist_roll_joint"),
                            ObservationType.JointVel("dq_right_shoulder_pitch_joint", xml_name="right_shoulder_pitch_joint"),
                            ObservationType.JointVel("dq_right_shoulder_roll_joint", xml_name="right_shoulder_roll_joint"),
                            ObservationType.JointVel("dq_right_shoulder_yaw_joint", xml_name="right_shoulder_yaw_joint"),
                            ObservationType.JointVel("dq_right_elbow_joint", xml_name="right_elbow_joint"),
                            ObservationType.JointVel("dq_right_wrist_roll_joint", xml_name="right_wrist_roll_joint"),]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Returns the default action specification of the environment.

        All actuators of the given specification are used, in the order in which they are defined.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: The names of the actuators that make up the default action space.

        """
        return [actuator.name for actuator in spec.actuators]

    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        """
        Reorients the arms to prevent collisions with the hips. Used when ``disable_arms`` is True.

        Args:
            spec (MjSpec): MuJoCo specification. It is modified in place.

        Returns:
            MjSpec: The modified MuJoCo specification.

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
        Returns the default path to the XML file of the Unitree G1 model (23 degrees of freedom).

        Returns:
            str: Path to the XML file.

        """
        return cls.get_model_path("unitree_g1", "g1_23dof.xml")

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body (the torso link) in the XML file.

        Returns:
            str: Name of the upper body.

        """
        return "torso_link"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the range of root heights in which the robot is considered healthy.

        This is only used when the ``HeightBasedTerminalStateHandler`` is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters.

        """
        return (0.5, 1.0)
