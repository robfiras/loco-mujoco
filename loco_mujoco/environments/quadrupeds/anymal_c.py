from typing import List, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class AnymalC(BaseRobotQuadruped):

    """
    Description
    ------------
    Mujoco environment of the Anymal C robot by ANYbotics.


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
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            camera_params (dict): Dictionary defining some of the camera parameters for visualization.
            observation_spec (List[ObservationType]): Observation specification.
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
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[ObservationType]: A list of observations.
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
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """
        action_spec = [
            "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        Returns:
            str: The default path to the xml file.
        """
        return cls.get_model_path("anybotics_anymal_c", "anymal_c.xml")

    @info_property
    def grf_size(self) -> int:
        """
        Returns the size of the ground force vector.

        Returns:
            int: The size of the ground force vector.
        """
        return 12

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml.

        Returns:
            str: The name of the upper body.
        """
        return self.root_body_name

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml.

        Returns:
            str: The name of the root body.
        """
        return "base"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root body in the Mujoco xml.

        Returns:
            str: The name of the free joint.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: The healthy range of the root height.
        """
        return (0.30, 1.0)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geometries.

        Returns:
            List[str]: The names of the foot geometries.
        """
        return ["LH", "RH", "LF", "RF"]