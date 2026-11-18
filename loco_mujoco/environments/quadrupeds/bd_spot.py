"""
Environment of the Boston Dynamics Spot quadruped robot (CPU/MuJoCo version).
"""

from typing import Union, List, Tuple
import numpy as np
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class BDSpot(BaseRobotQuadruped):

    """
    Environment of the Boston Dynamics Spot quadruped robot.

    Description
    -----------

    MuJoCo environment of Spot, a quadruped robot by Boston Dynamics. This class uses the CPU (MuJoCo) backend;
    :class:`~loco_mujoco.environments.quadrupeds.bd_spot_mjx.MjxBDSpot` is its Mjx counterpart.

    The model (``bd_spot/spot.xml``) has a floating base and 12 actuated joints: two hip joints and the knee joint of
    each of the four legs.

    By default, the observation has 35 dimensions: the height and the orientation quaternion of the floating base (its
    horizontal position is left out), the 12 joint positions, the 6D velocity of the floating base and the 12 joint
    velocities. The action has 12 dimensions, one per actuated joint. Unless ``control_type`` is given, a PD
    controller (``PDControl``) with a proportional gain of 200 and no derivative gain is used: the action is added to
    the joint positions of the standing pose (:attr:`init_qpos`) to obtain the position targets of the controller,
    without scaling to the joint limits. Both spaces are listed below and can be changed with the constructor
    arguments ``observation_spec`` and ``actuation_spec``.

    Unless another initial state handler or other parameters for it are given, an episode starts in the standing pose
    defined by :attr:`init_qpos` and :attr:`init_qvel`.

    Default Observation Space
    -------------------------

    ============ ======== ================ ==================================== ============================== ===
    Index in Obs Name     ObservationType  Min                                  Max                            Dim
    ============ ======== ================ ==================================== ============================== ===
    0 - 4        q_root   FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_fl_hx  JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_fl_hy  JointPos         [-0.898845]                          [2.29511]                      1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_fl_kn  JointPos         [-2.7929]                            [-0.254402]                    1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_fr_hx  JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_fr_hy  JointPos         [-0.898845]                          [2.24363]                      1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_fr_kn  JointPos         [-2.7929]                            [-0.255648]                    1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_hl_hx  JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_hl_hy  JointPos         [-0.898845]                          [2.29511]                      1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_hl_kn  JointPos         [-2.7929]                            [-0.247067]                    1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_hr_hx  JointPos         [-0.785398]                          [0.785398]                     1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_hr_hy  JointPos         [-0.898845]                          [2.29511]                      1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_hr_kn  JointPos         [-2.7929]                            [-0.248282]                    1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    17 - 22      dq_root  FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    23           dq_fl_hx JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    24           dq_fl_hy JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    25           dq_fl_kn JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    26           dq_fr_hx JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    27           dq_fr_hy JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    28           dq_fr_kn JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    29           dq_hl_hx JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    30           dq_hl_hy JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    31           dq_hl_kn JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    32           dq_hr_hx JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    33           dq_hr_hy JointVel         [-inf]                               [inf]                          1
    ------------ -------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_hr_kn JointVel         [-inf]                               [inf]                          1
    ============ ======== ================ ==================================== ============================== ===


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
    =============== ==== ===


    """

    mjx_enabled = False

    def __init__(self, spec=None, camera_params=None,
                 observation_spec=None, actuation_spec=None, **kwargs):
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec], optional): Model of the robot, either the path to a MuJoCo XML file or an
                ``MjSpec`` object. If None, the default XML file is used (see :meth:`get_default_xml_file_path`).
            camera_params (dict, optional): Parameters of the cameras used for visualization. If None, the
                ``follow`` camera is placed further away from the robot than by default.
            observation_spec (List[Observation], optional): Observations that make up the observation space, as
                :class:`~loco_mujoco.core.observations.base.Observation` objects or as dictionaries that are parsed
                with ``parse_observation_spec``. If None, the default observation space listed in the class docstring
                is used.
            actuation_spec (List[str], optional): Names of the actuators in the model that make up the action
                space. If None, the default action space listed in the class docstring is used.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.base.LocoEnv`.
                If ``control_type`` is not given, ``PDControl`` is used.
                If ``init_state_type`` is not given or is ``"DefaultInitialStateHandler"`` without
                ``init_state_params``, the robot is reset to the standing pose given by :attr:`init_qpos` and
                :attr:`init_qvel`.

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
            kwargs["control_params"] = dict(p_gain=200.0, d_gain=0.0, scale_action_to_jnt_limits=False,
                                            nominal_joint_positions=self.init_qpos[7:])

        # set init position: the robot's standing pose, unless the caller chose another handler or its params
        kwargs.setdefault("init_state_type", "DefaultInitialStateHandler")
        if kwargs["init_state_type"] == "DefaultInitialStateHandler" and kwargs.get("init_state_params") is None:
            kwargs["init_state_params"] = dict(qpos_init=self.init_qpos, qvel_init=self.init_qvel)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        if camera_params is None:
            # make the camera by default a bit higher
            camera_params = dict(follow=dict(distance=3.5, elevation=-20.0, azimuth=90.0))

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec,
                         camera_params=camera_params, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        """
        Returns the default observation specification of the environment.

        The default observation consists of the height and orientation of the floating base, the joint positions,
        the velocity of the floating base and the joint velocities.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[ObservationType]: The list of default observations.

        """

        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            ObservationType.FreeJointPosNoXY("q_root", xml_name="freejoint"),
            # --- Front Left ---
            ObservationType.JointPos("q_fl_hx", xml_name="fl_hx"),
            ObservationType.JointPos("q_fl_hy", xml_name="fl_hy"),
            ObservationType.JointPos("q_fl_kn", xml_name="fl_kn"),
            # --- Front Right ---
            ObservationType.JointPos("q_fr_hx", xml_name="fr_hx"),
            ObservationType.JointPos("q_fr_hy", xml_name="fr_hy"),
            ObservationType.JointPos("q_fr_kn", xml_name="fr_kn"),
            # --- Rear Left ---
            ObservationType.JointPos("q_hl_hx", xml_name="hl_hx"),
            ObservationType.JointPos("q_hl_hy", xml_name="hl_hy"),
            ObservationType.JointPos("q_hl_kn", xml_name="hl_kn"),
            # --- Rear Right ---
            ObservationType.JointPos("q_hr_hx", xml_name="hr_hx"),
            ObservationType.JointPos("q_hr_hy", xml_name="hr_hy"),
            ObservationType.JointPos("q_hr_kn", xml_name="hr_kn"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="freejoint"),
            # --- Front Left ---
            ObservationType.JointVel("dq_fl_hx", xml_name="fl_hx"),
            ObservationType.JointVel("dq_fl_hy", xml_name="fl_hy"),
            ObservationType.JointVel("dq_fl_kn", xml_name="fl_kn"),
            # --- Front Right ---
            ObservationType.JointVel("dq_fr_hx", xml_name="fr_hx"),
            ObservationType.JointVel("dq_fr_hy", xml_name="fr_hy"),
            ObservationType.JointVel("dq_fr_kn", xml_name="fr_kn"),
            # --- Rear Left ---
            ObservationType.JointVel("dq_hl_hx", xml_name="hl_hx"),
            ObservationType.JointVel("dq_hl_hy", xml_name="hl_hy"),
            ObservationType.JointVel("dq_hl_kn", xml_name="hl_kn"),
            # --- Rear Right ---
            ObservationType.JointVel("dq_hr_hx", xml_name="hr_hx"),
            ObservationType.JointVel("dq_hr_hy", xml_name="hr_hy"),
            ObservationType.JointVel("dq_hr_kn", xml_name="hr_kn"),
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
        action_spec = [
            "fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the XML file of the Spot model.

        Returns:
            str: Path to the XML file.

        """
        return cls.get_model_path("bd_spot", "spot.xml")

    @info_property
    def grf_size(self) -> int:
        """
        Returns the size of the ground reaction force vector (three components for each of the four feet).

        Returns:
            int: The size of the ground reaction force vector.

        """
        return 12

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
        return "body"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root in the XML file.

        Returns:
            str: Name of the free joint.

        """
        return "freejoint"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the range of root heights in which the robot is considered healthy.

        This is only used when the ``HeightBasedTerminalStateHandler`` is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters.

        """
        return (0.25, 1.0)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geoms in the XML file.

        Returns:
            List[str]: Names of the four foot geoms.

        """
        return ["HL", "HR", "FL", "FR"]

    @info_property
    def init_qpos(self) -> np.ndarray:
        """
        Returns the joint positions of the default initial state, a standing pose.

        Returns:
            np.ndarray: Array of shape ``(19,)`` with the position and orientation quaternion of the floating base
            followed by the 12 joint positions.

        """
        return np.array([0.0, 0.0, 0.46, 1.0, 0.0, 0.0, 0.0, 0.0, 1.04, -1.8, 0.0,
                         1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8])

    @info_property
    def init_qvel(self) -> np.ndarray:
        """
        Returns the joint velocities of the default initial state, which are all zero.

        Returns:
            np.ndarray: Array of shape ``(18,)`` with the velocity of the floating base followed by the 12 joint
            velocities.

        """
        return np.zeros(18)
