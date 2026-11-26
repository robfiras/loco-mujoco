"""
Environment of the ANYbotics ANYmal C quadruped robot (CPU/MuJoCo version).
"""

from typing import List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class AnymalC(BaseRobotQuadruped):

    """
    Environment of the ANYbotics ANYmal C quadruped robot.

    Description
    -----------

    MuJoCo environment of the ANYmal C, a quadruped robot by ANYbotics. This class uses the CPU (MuJoCo) backend;
    :class:`~loco_mujoco.environments.quadrupeds.anymal_c_mjx.MjxAnymalC` is its Mjx counterpart.

    The model (``anybotics_anymal_c/anymal_c.xml``) has a floating base and 12 actuated joints: hip
    abduction/adduction (HAA), hip flexion/extension (HFE) and knee flexion/extension (KFE) joint of each of the four
    legs.

    By default, the observation has 35 dimensions: the height and the orientation quaternion of the floating base (its
    horizontal position is left out), the 12 joint positions, the 6D velocity of the floating base and the 12 joint
    velocities. The action has 12 dimensions, one per actuated joint, each in ``[-1, 1]``. Unless ``control_type`` is
    given, a PD controller (``PDControl``) with a proportional gain of 100 and no derivative gain is used: an action
    in ``[-1, 1]`` is mapped linearly onto the range between the lower and the upper joint limit to obtain the
    position target of the controller. Both spaces are listed below and can be changed with the constructor arguments
    ``observation_spec`` and ``actuation_spec``.

    Default Observation Space
    -------------------------

    ============ ========= ================ ==================================== ============================== ===
    Index in Obs Name      ObservationType  Min                                  Max                            Dim
    ============ ========= ================ ==================================== ============================== ===
    0 - 4        q_root    FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_lf_haa  JointPos         [-0.72]                              [0.49]                         1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_lf_hfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_lf_kfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_rf_haa  JointPos         [-0.49]                              [0.72]                         1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_rf_hfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_rf_kfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_lh_haa  JointPos         [-0.72]                              [0.49]                         1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_lh_hfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_lh_kfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_rh_haa  JointPos         [-0.49]                              [0.72]                         1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_rh_hfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_rh_kfe  JointPos         [-9.42478]                           [9.42478]                      1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    17 - 22      dq_root   FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    23           dq_lf_haa JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    24           dq_lf_hfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    25           dq_lf_kfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    26           dq_rf_haa JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    27           dq_rf_hfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    28           dq_rf_kfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    29           dq_lh_haa JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    30           dq_lh_hfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    31           dq_lh_kfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    32           dq_rh_haa JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    33           dq_rh_hfe JointVel         [-inf]                               [inf]                          1
    ------------ --------- ---------------- ------------------------------------ ------------------------------ ---
    34           dq_rh_kfe JointVel         [-inf]                               [inf]                          1
    ============ ========= ================ ==================================== ============================== ===

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
            kwargs["control_params"] = dict(p_gain=100.0, d_gain=0.0)

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
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            # --- Front Left ---
            ObservationType.JointPos("q_lf_haa", xml_name="LF_HAA"),
            ObservationType.JointPos("q_lf_hfe", xml_name="LF_HFE"),
            ObservationType.JointPos("q_lf_kfe", xml_name="LF_KFE"),
            # --- Front Right ---
            ObservationType.JointPos("q_rf_haa", xml_name="RF_HAA"),
            ObservationType.JointPos("q_rf_hfe", xml_name="RF_HFE"),
            ObservationType.JointPos("q_rf_kfe", xml_name="RF_KFE"),
            # --- Rear Left ---
            ObservationType.JointPos("q_lh_haa", xml_name="LH_HAA"),
            ObservationType.JointPos("q_lh_hfe", xml_name="LH_HFE"),
            ObservationType.JointPos("q_lh_kfe", xml_name="LH_KFE"),
            # --- Rear Right ---
            ObservationType.JointPos("q_rh_haa", xml_name="RH_HAA"),
            ObservationType.JointPos("q_rh_hfe", xml_name="RH_HFE"),
            ObservationType.JointPos("q_rh_kfe", xml_name="RH_KFE"),

            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            # --- Front Left ---
            ObservationType.JointVel("dq_lf_haa", xml_name="LF_HAA"),
            ObservationType.JointVel("dq_lf_hfe", xml_name="LF_HFE"),
            ObservationType.JointVel("dq_lf_kfe", xml_name="LF_KFE"),
            # --- Front Right ---
            ObservationType.JointVel("dq_rf_haa", xml_name="RF_HAA"),
            ObservationType.JointVel("dq_rf_hfe", xml_name="RF_HFE"),
            ObservationType.JointVel("dq_rf_kfe", xml_name="RF_KFE"),
            # --- Rear Left ---
            ObservationType.JointVel("dq_lh_haa", xml_name="LH_HAA"),
            ObservationType.JointVel("dq_lh_hfe", xml_name="LH_HFE"),
            ObservationType.JointVel("dq_lh_kfe", xml_name="LH_KFE"),
            # --- Rear Right ---
            ObservationType.JointVel("dq_rh_haa", xml_name="RH_HAA"),
            ObservationType.JointVel("dq_rh_hfe", xml_name="RH_HFE"),
            ObservationType.JointVel("dq_rh_kfe", xml_name="RH_KFE"),
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
            "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the XML file of the ANYmal C model.

        Returns:
            str: Path to the XML file.

        """
        return cls.get_model_path("anybotics_anymal_c", "anymal_c.xml")

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
        return "base"

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
        return (0.30, 1.0)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geoms in the XML file.

        Returns:
            List[str]: Names of the four foot geoms.

        """
        return ["LH", "RH", "LF", "RF"]