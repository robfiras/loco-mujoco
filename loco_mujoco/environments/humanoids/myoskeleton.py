import os
from typing import Union, List, Tuple, Dict

import mujoco
from mujoco import MjSpec
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils import info_property
from loco_mujoco import PATH_TO_CUSTOM_MODELS


class MyoSkeleton(LocoEnv):
    """
    Description
    ------------

    MuJoCo simulation of the MyoSkeleton with one position controller per joint. This is a biomechanical
    humanoid model with many degrees of freedom (151 joints) to accurately simulate human movement. The model
    extends the simple humanoid by a spine, hands with fingers, realistic knees, and feet with toes.


    Default Observation Space
    -------------------------

    ============ ============= ================ =========================================================================== ======================================================================== ===
    Index in Obs Name          ObservationType  Min                                                                         Max                                                                      Dim
    ============ ============= ================ =========================================================================== ======================================================================== ===
    0 - 4        q_free_joint  FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]                                              [inf, inf, inf, inf, inf]                                                5
    ------------ ------------- ---------------- --------------------------------------------------------------------------- ------------------------------------------------------------------------ ---
    5 - 155      q_all_pos     JointPosArray    [-1.22173048, -0.43633231, -0.9791297 ... -0.0408267, -0.0227731, -1.79241] [0.45378561, 0.43633231, 0.97912971 ... -0.0108281, 0.0524192, 0.010506] 151
    ------------ ------------- ---------------- --------------------------------------------------------------------------- ------------------------------------------------------------------------ ---
    156 - 161    dq_free_joint FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf]                                        [inf, inf, inf, inf, inf, inf]                                           6
    ------------ ------------- ---------------- --------------------------------------------------------------------------- ------------------------------------------------------------------------ ---
    162 - 312    dq_all_vel    JointVelArray    [-inf, -inf, -inf ... -inf, -inf, -inf]                                     [inf, inf, inf ... inf, inf, inf]                                        151
    ============ ============= ================ =========================================================================== ======================================================================== ===


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
    --------------- ---- ---
    30              -1.0 1.0
    --------------- ---- ---
    31              -1.0 1.0
    --------------- ---- ---
    32              -1.0 1.0
    --------------- ---- ---
    33              -1.0 1.0
    --------------- ---- ---
    34              -1.0 1.0
    --------------- ---- ---
    35              -1.0 1.0
    --------------- ---- ---
    36              -1.0 1.0
    --------------- ---- ---
    37              -1.0 1.0
    --------------- ---- ---
    38              -1.0 1.0
    --------------- ---- ---
    39              -1.0 1.0
    --------------- ---- ---
    40              -1.0 1.0
    --------------- ---- ---
    41              -1.0 1.0
    --------------- ---- ---
    42              -1.0 1.0
    --------------- ---- ---
    43              -1.0 1.0
    --------------- ---- ---
    44              -1.0 1.0
    --------------- ---- ---
    45              -1.0 1.0
    --------------- ---- ---
    46              -1.0 1.0
    --------------- ---- ---
    47              -1.0 1.0
    --------------- ---- ---
    48              -1.0 1.0
    --------------- ---- ---
    49              -1.0 1.0
    --------------- ---- ---
    50              -1.0 1.0
    --------------- ---- ---
    51              -1.0 1.0
    --------------- ---- ---
    52              -1.0 1.0
    --------------- ---- ---
    53              -1.0 1.0
    --------------- ---- ---
    54              -1.0 1.0
    --------------- ---- ---
    55              -1.0 1.0
    --------------- ---- ---
    56              -1.0 1.0
    --------------- ---- ---
    57              -1.0 1.0
    --------------- ---- ---
    58              -1.0 1.0
    --------------- ---- ---
    59              -1.0 1.0
    --------------- ---- ---
    60              -1.0 1.0
    --------------- ---- ---
    61              -1.0 1.0
    --------------- ---- ---
    62              -1.0 1.0
    --------------- ---- ---
    63              -1.0 1.0
    --------------- ---- ---
    64              -1.0 1.0
    --------------- ---- ---
    65              -1.0 1.0
    --------------- ---- ---
    66              -1.0 1.0
    --------------- ---- ---
    67              -1.0 1.0
    --------------- ---- ---
    68              -1.0 1.0
    --------------- ---- ---
    69              -1.0 1.0
    --------------- ---- ---
    70              -1.0 1.0
    --------------- ---- ---
    71              -1.0 1.0
    --------------- ---- ---
    72              -1.0 1.0
    --------------- ---- ---
    73              -1.0 1.0
    --------------- ---- ---
    74              -1.0 1.0
    --------------- ---- ---
    75              -1.0 1.0
    --------------- ---- ---
    76              -1.0 1.0
    --------------- ---- ---
    77              -1.0 1.0
    --------------- ---- ---
    78              -1.0 1.0
    --------------- ---- ---
    79              -1.0 1.0
    --------------- ---- ---
    80              -1.0 1.0
    --------------- ---- ---
    81              -1.0 1.0
    --------------- ---- ---
    82              -1.0 1.0
    --------------- ---- ---
    83              -1.0 1.0
    --------------- ---- ---
    84              -1.0 1.0
    --------------- ---- ---
    85              -1.0 1.0
    --------------- ---- ---
    86              -1.0 1.0
    --------------- ---- ---
    87              -1.0 1.0
    --------------- ---- ---
    88              -1.0 1.0
    --------------- ---- ---
    89              -1.0 1.0
    --------------- ---- ---
    90              -1.0 1.0
    --------------- ---- ---
    91              -1.0 1.0
    --------------- ---- ---
    92              -1.0 1.0
    --------------- ---- ---
    93              -1.0 1.0
    --------------- ---- ---
    94              -1.0 1.0
    --------------- ---- ---
    95              -1.0 1.0
    --------------- ---- ---
    96              -1.0 1.0
    --------------- ---- ---
    97              -1.0 1.0
    --------------- ---- ---
    98              -1.0 1.0
    --------------- ---- ---
    99              -1.0 1.0
    --------------- ---- ---
    100             -1.0 1.0
    --------------- ---- ---
    101             -1.0 1.0
    --------------- ---- ---
    102             -1.0 1.0
    --------------- ---- ---
    103             -1.0 1.0
    --------------- ---- ---
    104             -1.0 1.0
    --------------- ---- ---
    105             -1.0 1.0
    --------------- ---- ---
    106             -1.0 1.0
    --------------- ---- ---
    107             -1.0 1.0
    --------------- ---- ---
    108             -1.0 1.0
    --------------- ---- ---
    109             -1.0 1.0
    --------------- ---- ---
    110             -1.0 1.0
    --------------- ---- ---
    111             -1.0 1.0
    --------------- ---- ---
    112             -1.0 1.0
    --------------- ---- ---
    113             -1.0 1.0
    --------------- ---- ---
    114             -1.0 1.0
    --------------- ---- ---
    115             -1.0 1.0
    --------------- ---- ---
    116             -1.0 1.0
    --------------- ---- ---
    117             -1.0 1.0
    --------------- ---- ---
    118             -1.0 1.0
    --------------- ---- ---
    119             -1.0 1.0
    --------------- ---- ---
    120             -1.0 1.0
    --------------- ---- ---
    121             -1.0 1.0
    --------------- ---- ---
    122             -1.0 1.0
    --------------- ---- ---
    123             -1.0 1.0
    --------------- ---- ---
    124             -1.0 1.0
    --------------- ---- ---
    125             -1.0 1.0
    --------------- ---- ---
    126             -1.0 1.0
    --------------- ---- ---
    127             -1.0 1.0
    --------------- ---- ---
    128             -1.0 1.0
    --------------- ---- ---
    129             -1.0 1.0
    --------------- ---- ---
    130             -1.0 1.0
    --------------- ---- ---
    131             -1.0 1.0
    --------------- ---- ---
    132             -1.0 1.0
    --------------- ---- ---
    133             -1.0 1.0
    --------------- ---- ---
    134             -1.0 1.0
    --------------- ---- ---
    135             -1.0 1.0
    --------------- ---- ---
    136             -1.0 1.0
    --------------- ---- ---
    137             -1.0 1.0
    --------------- ---- ---
    138             -1.0 1.0
    --------------- ---- ---
    139             -1.0 1.0
    --------------- ---- ---
    140             -1.0 1.0
    --------------- ---- ---
    141             -1.0 1.0
    --------------- ---- ---
    142             -1.0 1.0
    --------------- ---- ---
    143             -1.0 1.0
    --------------- ---- ---
    144             -1.0 1.0
    --------------- ---- ---
    145             -1.0 1.0
    --------------- ---- ---
    146             -1.0 1.0
    --------------- ---- ---
    147             -1.0 1.0
    --------------- ---- ---
    148             -1.0 1.0
    --------------- ---- ---
    149             -1.0 1.0
    --------------- ---- ---
    150             -1.0 1.0
    =============== ==== ===

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_fingers: bool = True,
                 spec: MjSpec = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_fingers (bool): If True, the fingers are disabled.
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            observation_spec (List[Observation]): Observation specification.
            actuation_spec (List[str]): Action specification.
            **kwargs: Additional arguments

        """
        
        self._disable_fingers = disable_fingers

        if spec is None:
            spec = self.get_default_xml_file_path()

            # check if file exists, if not exit
            if not os.path.exists(spec):
                print(
                    "MyoSkeleton model not initialized. Please run \"loco-mujoco-myomodel-init\" to accept the license "
                    "and download the model. Exiting...")
                exit()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # apply changes to the MjSpec
        spec = self._apply_spec_changes(spec)

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

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_observation_specification(self, spec: MjSpec) -> List[Observation]:
        """
        Getter for the observation space specification. This function reads all joint names from the xml and adds
        the prefix ``q_`` for the joint positions and ``dq_`` for the joint velocities. It also adds the free joint
        position (disregarding the x and y position) and velocity.

        Returns:
            List[Observation]: List of observations.

        """
        # get all joint names except the root
        j_names = [j.name for j in spec.joints if j.name != self.root_free_joint_xml_name]

        # build observation spec
        observation_spec = []

        # add free joint observation
        observation_spec.append(ObservationType.FreeJointPosNoXY("q_free_joint", self.root_free_joint_xml_name))

        # add all joint positions
        observation_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # add free joint velocities
        observation_spec.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))

        # add all joint velocities
        observation_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        return observation_spec

    def _get_action_specification(self, spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification. This function adds all actuator names found in the spec, which
        are the ones added in the _add_actuators method.

        Returns:
            List[str]: A list of tuples containing the specification of each action
            space entry.

        """
        action_spec = []
        for a in spec.actuators:
            action_spec.append(a.name)
        return action_spec

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        This function reads the original myo_model spec and applies some changes to make it align with LocoMuJoCo.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        def _copy_attributes(src, dst, skip=None):
            skip = set(skip or [])
            skip |= {"element", "alt"}
            for attr in dir(src):
                if attr.startswith("__") or attr in skip or callable(getattr(src, attr)):
                    continue
                try:
                    setattr(dst, attr, getattr(src, attr))
                except (TypeError, AttributeError):
                    pass

        # remove floor and add ground plane
        for g in spec.geoms:
            if g.name == "floor":
                spec.delete(g)

        # remove old lights
        for b in spec.bodies:
            for l in b.lights:
                spec.delete(l)

        # load common specs
        scene_spec = mujoco.MjSpec.from_file(self.get_model_path("common", "scene.xml"))

        # add all textures, materials, geoms and lights (add then copy attributes)
        for t in scene_spec.textures:
            new_t = spec.add_texture()
            _copy_attributes(t, new_t)
        for m in scene_spec.materials:
            new_m = spec.add_material()
            _copy_attributes(m, new_m)
        for g in scene_spec.geoms:
            new_g = spec.worldbody.add_geom()
            _copy_attributes(g, new_g)
        for l in scene_spec.lights:
            new_l = spec.worldbody.add_light()
            _copy_attributes(l, new_l)

        # use default scene visuals
        spec.visual = scene_spec.visual

        # add mimic sites
        for body_name, site_name in self.body2sites_for_mimic.items():
            b = [body for body in spec.bodies if body.name == body_name][0]
            pos = [0.0, 0.0, 0.0]
            # todo: can not load mimic sites attributes for now, so I add them manually
            b.add_site(name=site_name, group=4, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.075, 0.05, 0.025],
                       rgba=[1.0, 0.0, 0.0, 0.5], pos=pos)

        # add spot light
        for b in spec.bodies:
            if b.name == "pelvis":
                b.add_light(name="spotlight", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[0, 50, -2], dir=[0, -1, 0])

        if self._disable_fingers:
            for j in spec.joints:
                if "finger" in self.finger_and_hand_joints:
                    spec.delete(j)

        # add actuators
        spec = self._add_actuators(spec)

        return spec

    def _add_actuators(self, spec: MjSpec) -> MjSpec:
        """
        Adds a generic actuator to each joint.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """
        max_joint_forces = dict(L5_S1_Flex_Ext=200,
                                L5_S1_Lat_Bending=200,
                                L5_S1_axial_rotation=200,
                                L4_L5_Flex_Ext=200,
                                L4_L5_Lat_Bending=200,
                                L4_L5_axial_rotation=200,
                                L3_L4_Flex_Ext=200,
                                L3_L4_Lat_Bending=200,
                                L3_L4_axial_rotation=200,
                                L2_L3_Flex_Ext=200,
                                L2_L3_Lat_Bending=200,
                                L2_L3_axial_rotation=200,
                                L1_L2_Flex_Ext=200,
                                L1_L2_Lat_Bending=200,
                                L1_L2_axial_rotation=200,
                                L1_T12_Flex_Ext=200,
                                L1_T12_Lat_Bending=200,
                                L1_T12_axial_rotation=200,
                                c7_c6_FE=50,
                                c7_c6_LB=50,
                                c7_c6_AR=50,
                                c6_c5_FE=50,
                                c6_c5_LB=50,
                                c6_c5_AR=50,
                                c5_c4_FE=50,
                                c5_c4_LB=50,
                                c5_c4_AR=50,
                                c4_c3_FE=50,
                                c4_c3_LB=50,
                                c4_c3_AR=50,
                                c3_c2_FE=50,
                                c3_c2_LB=50,
                                c3_c2_AR=50,
                                c2_c1_FE=50,
                                c2_c1_LB=50,
                                c2_c1_AR=50,
                                c1_skull_FE=50,
                                c1_skull_LB=50,
                                c1_skull_AR=50,
                                skull_FE=50,
                                skull_LB=50,
                                skull_AR=50,
                                sternoclavicular_r2_r=80,
                                sternoclavicular_r3_r=80,
                                unrotscap_r3_r=80,
                                unrotscap_r2_r=80,
                                acromioclavicular_r2_r=80,
                                acromioclavicular_r3_r=80,
                                acromioclavicular_r1_r=80,
                                unrothum_r1_r=80,
                                unrothum_r3_r=80,
                                unrothum_r2_r=80,
                                elv_angle_r=80,
                                shoulder_elv_r=80,
                                shoulder1_r2_r=80,
                                shoulder_rot_r=80,
                                elbow_flex_r=80,
                                pro_sup=80,
                                deviation=80,
                                flexion_r=80,
                                sternoclavicular_r2_l=80,
                                sternoclavicular_r3_l=80,
                                unrotscap_r3_l=80,
                                unrotscap_r2_l=80,
                                acromioclavicular_r2_l=80,
                                acromioclavicular_r3_l=80,
                                acromioclavicular_r1_l=80,
                                unrothum_r1_l=80,
                                unrothum_r3_l=80,
                                unrothum_r2_l=80,
                                elv_angle_l=80,
                                shoulder_elv_l=80,
                                shoulder1_r2_l=80,
                                shoulder_rot_l=80,
                                elbow_flex_l=80,
                                pro_sup_l=80,
                                deviation_l=80,
                                flexion_l=80,
                                hip_flexion_r=200,
                                hip_adduction_r=200,
                                hip_rotation_r=200,
                                knee_angle_r=200,
                                knee_angle_r_rotation2=20,
                                knee_angle_r_rotation3=20,
                                ankle_angle_r=200,
                                subtalar_angle_r=200,
                                mtp_angle_r=200,
                                knee_angle_r_beta_rotation1=20,
                                hip_flexion_l=200,
                                hip_adduction_l=200,
                                hip_rotation_l=200,
                                knee_angle_l=200,
                                knee_angle_l_rotation2=20,
                                knee_angle_l_rotation3=20,
                                ankle_angle_l=200,
                                subtalar_angle_l=200,
                                mtp_angle_l=200,
                                knee_angle_l_beta_rotation1=20)

        for joint in spec.joints:
            # add an actuator for every joint except the pelvis
            if self.root_free_joint_xml_name not in joint.name:
                max_force = max_joint_forces[joint.name] if joint.name in max_joint_forces.keys() else 50
                spec.add_actuator(name="act_" + joint.name, target=joint.name, ctrlrange=[-max_force, max_force],
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrllimited=True)

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        """
        return (PATH_TO_CUSTOM_MODELS / "myo_model" / "myoskeleton" / "myoskeleton.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml file.

        """
        return "thoracic_spine"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint in the Mujoco xml file.

        """
        return "myoskeleton_root"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml file.

        """
        return "myoskeleton_root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        """
        return (0.6, 1.5)

    @info_property
    def body2sites_for_mimic(self) -> Dict[str, str]:
        """
        Returns a dictionary that maps body names to mimic site names.

        Returns:
            Dict[str, str]: Mapping from body names to mimic site names.

        """
        body2sitemimic = {
            "thoracic_spine": "upper_body_mimic",
            "skull": "head_mimic",  # Adding head mimic (likely attached to the skull)
            "pelvis": "pelvis_mimic",
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "calcn_l": "left_foot_mimic",
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "calcn_r": "right_foot_mimic"
        }

        return body2sitemimic
    
    @info_property
    def finger_and_hand_joints(self) -> List[str]:
        """
        Returns the names of the finger and hand joints.

        Returns:
            List[str]: List of finger and hand joint names.

        """
        finger_hand_joints = [
            # Thumb (Right)
            "cmc_flexion_r", "cmc_abduction_r",
            "mp_flexion_r",
            "ip_flexion_r",

            # Index Finger (Right)
            "mcp2_flexion_r", "mcp2_abduction_r",
            "pm2_flexion_r",
            "md2_flexion_r",

            # Middle Finger (Right)
            "mcp3_flexion_r", "mcp3_abduction_r",
            "pm3_flexion_r",
            "md3_flexion_r",

            # Ring Finger (Right)
            "mcp4_flexion_r", "mcp4_abduction_r",
            "pm4_flexion_r",
            "md4_flexion_r",

            # Little Finger (Right)
            "mcp5_flexion_r", "mcp5_abduction_r",
            "pm5_flexion_r",
            "md5_flexion_r",

            # Thumb (Left)
            "cmc_flexion_l", "cmc_abduction_l",
            "mp_flexion_l",
            "ip_flexion_l",

            # Index Finger (Left)
            "mcp2_flexion_l", "mcp2_abduction_l",
            "pm2_flexion_l",
            "md2_flexion_l",

            # Middle Finger (Left)
            "mcp3_flexion_l", "mcp3_abduction_l",
            "pm3_flexion_l",
            "md3_flexion_l",

            # Ring Finger (Left)
            "mcp4_flexion_l", "mcp4_abduction_l",
            "pm4_flexion_l",
            "md4_flexion_l",

            # Little Finger (Left)
            "mcp5_flexion_l", "mcp5_abduction_l",
            "pm5_flexion_l",
            "md5_flexion_l",
        ]

        return finger_hand_joints

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Returns a list of all mimic sites.

        """
        return list(self.body2sites_for_mimic.values())

    @info_property
    def goal_visualization_arrow_offset(self) -> List[float]:
        """
        Returns the offset for the goal visualization arrow.

        """
        return [0, 0, 0.4]
