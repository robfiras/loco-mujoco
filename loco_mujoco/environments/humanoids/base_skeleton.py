"""
Base class of the humanoid skeleton environments.
"""

from typing import Union, List, Tuple
import numpy as np
import mujoco
from mujoco import MjSpec

from loco_mujoco.core import ObservationType
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils import info_property
from loco_mujoco.core.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData, TrajectoryHandler
from loco_mujoco.smpl.retargeting import extend_motion


class BaseSkeleton(LocoEnv):
    """
    Base class for the environments of the humanoid skeleton model.

    The skeleton is a model of the human body with a floating base (pelvis), three lumbar joints, seven
    joints per arm and seven joints per leg. The subclasses
    :class:`~loco_mujoco.environments.humanoids.skeletons.SkeletonTorque` and
    :class:`~loco_mujoco.environments.humanoids.skeletons.SkeletonMuscle` select the actuation (torque
    actuators or muscles) and define the action space, while this class defines the observation space and
    the modifications of the model that both share:

    - Box feet (``use_box_feet``): each foot is replaced by a box for simplified contacts, and the subtalar
      and toe joints are removed.
    - Disabled arms (``disable_arms``): all arm joints and their actuators are removed.
    - Body scaling (``scaling``): the kinematics and dynamics of the model are scaled to simulate humans of
      different sizes. Trajectories loaded into a scaled environment are scaled accordingly.

    By default, the observation consists of the pose of the floating base without its x-y position (height
    and orientation quaternion), all joint positions, the linear and angular velocity of the floating base,
    and all joint velocities.

    .. note:: This class is not the base for the MyoSkeleton environment.

    """

    mjx_enabled = False

    def __init__(self,
                 use_muscles: bool = False,
                 use_box_feet: bool = True,
                 disable_arms: bool = False,
                 scaling: float = 1.0,
                 alpha_box_feet: float = 0.5,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[ObservationType] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            use_muscles (bool): If True, muscle actuators will be used, else torque actuators will be used.
                This is set by the subclasses.
            use_box_feet (bool): If True, boxes are used as feet (for simplification). The original foot
                geometries are made non-colliding, and the subtalar and toe joints are removed together with
                their observations and, for the torque-actuated model, their actuators. Required for the Mjx
                variants with reduced contacts.
            disable_arms (bool): If True, all arm joints are removed and the respective
                observations and actuators are removed from the observation and action specification. The
                arms are then fixed in a new default orientation.
            scaling (float): Scaling factor for the kinematics and dynamics of the humanoid model. Lengths are
                scaled linearly, masses with the third and inertias with the fifth power of the factor, and
                actuator gears and peak muscle forces with the second power.
            alpha_box_feet (float): Alpha (opacity) value of the color of the boxes, which might be added as
                feet.
            spec (Union[str, MjSpec]): Model to simulate, either a path to a MuJoCo XML file or an ``MjSpec``
                object. If None, the default model of this environment is used.
            observation_spec (List[ObservationType]): Observations that make up the observation vector, given
                as observation objects or as dictionaries with a ``type`` key naming the observation type and
                its parameters as the remaining keys. If None, the default observation space is used.
            actuation_spec (List[str]): Names of the actuators in the model that are controlled by the action.
                If None, the default action space is used.
            **kwargs: Additional arguments passed to :class:`~loco_mujoco.environments.base.LocoEnv`. The
                extra key ``mjx_reduced_contacts`` (bool, default True) only affects the Mjx variants: if True,
                the contacts are reduced by ``_modify_spec_for_mjx``; if False, the collision geometries of the
                model are kept.

        """

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        self.scaling = scaling
        if scaling != 1.0:
            spec = self.scale_body(spec, use_muscles)

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_muscles = use_muscles
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_spec_modifications()

        if self._use_box_feet or self._disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in motors_to_remove]

            spec = self._delete_from_spec(spec, joints_to_remove,
                                          motors_to_remove, equ_constr_to_remove)
            if self._use_box_feet:
                spec = self._add_box_feet_to_spec(spec, alpha_box_feet)

            if self._disable_arms:
                spec = self._reorient_arms(spec)

        # mjx_reduced_contacts=False keeps the robot's own collision geoms on Mjx/MjWarp instead of the
        # simplified feet-only contacts of _modify_spec_for_mjx
        if kwargs.pop("mjx_reduced_contacts", True) and self.mjx_enabled:
            assert use_box_feet
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, motors, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: Names of the joints to remove, names of the motors to
            remove, and names of the equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []
        if self._use_box_feet:
            joints_to_remove += ["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            if not self._use_muscles:
                motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]

        if self._disable_arms:
            joints_to_remove += ["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                                 "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                                 "wrist_flex_l", "wrist_dev_l"]
            motors_to_remove += ["mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                                 "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                                 "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                                 "mot_wrist_flex_l", "mot_wrist_dev_l"]
            equ_constr_to_remove += ["wrist_flex_r_constraint", "wrist_dev_r_constraint",
                                     "wrist_flex_l_constraint", "wrist_dev_l_constraint"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[ObservationType]: List of observations.

        """

        observation_spec = [  # ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            # --- lower limb right ---
                            ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointPos("q_subtalar_angle_r", xml_name="subtalar_angle_r"),
                            ObservationType.JointPos("q_mtp_angle_r", xml_name="mtp_angle_r"),
                            # --- lower limb left ---
                            ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),
                            ObservationType.JointPos("q_subtalar_angle_l", xml_name="subtalar_angle_l"),
                            ObservationType.JointPos("q_mtp_angle_l", xml_name="mtp_angle_l"),
                            # --- lumbar ---
                            ObservationType.JointPos("q_lumbar_extension", xml_name="lumbar_extension"),
                            ObservationType.JointPos("q_lumbar_bending", xml_name="lumbar_bending"),
                            ObservationType.JointPos("q_lumbar_rotation", xml_name="lumbar_rotation"),
                            # --- upper body right ---
                            ObservationType.JointPos("q_arm_flex_r", xml_name="arm_flex_r"),
                            ObservationType.JointPos("q_arm_add_r", xml_name="arm_add_r"),
                            ObservationType.JointPos("q_arm_rot_r", xml_name="arm_rot_r"),
                            ObservationType.JointPos("q_elbow_flex_r", xml_name="elbow_flex_r"),
                            ObservationType.JointPos("q_pro_sup_r", xml_name="pro_sup_r"),
                            ObservationType.JointPos("q_wrist_flex_r", xml_name="wrist_flex_r"),
                            ObservationType.JointPos("q_wrist_dev_r", xml_name="wrist_dev_r"),
                            # --- upper body left ---
                            ObservationType.JointPos("q_arm_flex_l", xml_name="arm_flex_l"),
                            ObservationType.JointPos("q_arm_add_l", xml_name="arm_add_l"),
                            ObservationType.JointPos("q_arm_rot_l", xml_name="arm_rot_l"),
                            ObservationType.JointPos("q_elbow_flex_l", xml_name="elbow_flex_l"),
                            ObservationType.JointPos("q_pro_sup_l", xml_name="pro_sup_l"),
                            ObservationType.JointPos("q_wrist_flex_l", xml_name="wrist_flex_l"),
                            ObservationType.JointPos("q_wrist_dev_l", xml_name="wrist_dev_l"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            # --- lower limb right ---
                            ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
                            ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
                            ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
                            ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
                            ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
                            ObservationType.JointVel("dq_subtalar_angle_r", xml_name="subtalar_angle_r"),
                            ObservationType.JointVel("dq_mtp_angle_r", xml_name="mtp_angle_r"),
                            # --- lower limb left ---
                            ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
                            ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
                            ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
                            ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
                            ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l"),
                            ObservationType.JointVel("dq_subtalar_angle_l", xml_name="subtalar_angle_l"),
                            ObservationType.JointVel("dq_mtp_angle_l", xml_name="mtp_angle_l"),
                            # --- lumbar ---
                            ObservationType.JointVel("dq_lumbar_extension", xml_name="lumbar_extension"),
                            ObservationType.JointVel("dq_lumbar_bending", xml_name="lumbar_bending"),
                            ObservationType.JointVel("dq_lumbar_rotation", xml_name="lumbar_rotation"),
                            # --- upper body right ---
                            ObservationType.JointVel("dq_arm_flex_r", xml_name="arm_flex_r"),
                            ObservationType.JointVel("dq_arm_add_r", xml_name="arm_add_r"),
                            ObservationType.JointVel("dq_arm_rot_r", xml_name="arm_rot_r"),
                            ObservationType.JointVel("dq_elbow_flex_r", xml_name="elbow_flex_r"),
                            ObservationType.JointVel("dq_pro_sup_r", xml_name="pro_sup_r"),
                            ObservationType.JointVel("dq_wrist_flex_r", xml_name="wrist_flex_r"),
                            ObservationType.JointVel("dq_wrist_dev_r", xml_name="wrist_dev_r"),
                            # --- upper body left ---
                            ObservationType.JointVel("dq_arm_flex_l", xml_name="arm_flex_l"),
                            ObservationType.JointVel("dq_arm_add_l", xml_name="arm_add_l"),
                            ObservationType.JointVel("dq_arm_rot_l", xml_name="arm_rot_l"),
                            ObservationType.JointVel("dq_elbow_flex_l", xml_name="elbow_flex_l"),
                            ObservationType.JointVel("dq_pro_sup_l", xml_name="pro_sup_l"),
                            ObservationType.JointVel("dq_wrist_flex_l", xml_name="wrist_flex_l"),
                            ObservationType.JointVel("dq_wrist_dev_l", xml_name="wrist_dev_l")]

        return observation_spec

    def _add_box_feet_to_spec(self, spec: MjSpec,
                              alpha_box_feet: float) -> MjSpec:
        """
        Adds box feet to Mujoco spec and makes old feet non-collidable.

        Args:
            spec (MjSpec): Mujoco specification.
            alpha_box_feet (float): Alpha (opacity) value of the color of the boxes.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        # find foot and attach box
        toe_l = [body for body in spec.bodies if body.name == "toes_l"][0]
        size = np.array([0.112, 0.03, 0.05]) * self.scaling
        pos = np.array([-0.09, 0.019, 0.0]) * self.scaling
        toe_l.add_geom(name="foot_box_l", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                       rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
        toe_r = [body for body in spec.bodies if body.name == "toes_r"][0]
        toe_r.add_geom(name="foot_box_r", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                       rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_geoms = ["r_foot", "r_bofoot", "l_foot", "l_bofoot"]
        for g in spec.geoms:
            if g.name in foot_geoms:
                g.contype = 0
                g.conaffinity = 0

        return spec

    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        """
        Reorients the arms of a humanoid model given its Mujoco specification.

        This sets a fixed orientation of the upper arms and forearms and is applied when the arm joints are
        removed.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        h = [body for body in spec.bodies if body.name == "humerus_l"][0]
        h.quat = [1.0, -0.1, -1.0, -0.1]
        h = [body for body in spec.bodies if body.name == "ulna_l"][0]
        h.quat = [1.0, 0.6, 0.0, 0.0]
        h = [body for body in spec.bodies if body.name == "humerus_r"][0]
        h.quat = [1.0, 0.1, 1.0, -0.1]
        h = [body for body in spec.bodies if body.name == "ulna_r"][0]
        h.quat = [1.0, -0.6, 0.0, 0.0]

        return spec

    def scale_body(self, mjspec: MjSpec,
                   use_muscles: bool) -> MjSpec:
        """
        Scales the kinematics and dynamics of the humanoid model by the factor ``self.scaling``.

        Meshes (except the head) and body positions are scaled linearly, body masses with the third power and
        body inertias with the fifth power of the scaling factor. The gears of the torque actuators and the
        peak isometric forces of the muscles are scaled with the second power. If muscles are used, the
        positions of the sites are scaled as well. The specification is modified in place.

        Args:
            mjspec (MjSpec): Mujoco specification.
            use_muscles (bool): If True, muscle actuators will be scaled, else torque actuators will be scaled.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        body_scaling = self.scaling
        head_geoms = ["hat_skull", "hat_jaw", "hat_ribs_cap"]

        # scale meshes
        for mesh in mjspec.meshes:
            if mesh.name not in head_geoms: # don't scale head
                mesh.scale *= body_scaling

        # change position of head
        for geom in mjspec.geoms:
            if geom.name in head_geoms:
                geom.pos = [0.0, -0.5 * (1 - body_scaling), 0.0]

        # scale bodies
        for body in mjspec.bodies:
            body.pos *= body_scaling
            body.mass *= body_scaling ** 3
            body.fullinertia *= body_scaling ** 5
            assert np.array_equal(body.fullinertia[3:], np.zeros(3)), "Some of the off-diagonal elements of the " \
                                                                      "inertia matrix are not zero! Scaling is " \
                                                                      "not done correctly. Double-Check!"

        # scale actuators
        if use_muscles:
            for site in mjspec.sites:
                site.pos *= body_scaling

            for actuator in mjspec.actuators:
                if "mot" not in actuator.name:
                    # peak isometric force of a muscle: third muscle parameter (the `force`
                    # attribute in the XML), stored in both the gain and the bias parameters
                    actuator.gainprm[2] *= body_scaling ** 2
                    actuator.biasprm[2] *= body_scaling ** 2
                else:
                    actuator.gear *= body_scaling ** 2
        else:
            for actuator in mjspec.actuators:
                actuator.gear *= body_scaling ** 2

        return mjspec

    def process_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        Processes a trajectory and loads it into the environment.

        If ``scaling != 1.0``, the trajectory is first processed by the parent class, then the position and
        the linear velocity of the free joint are multiplied by the scaling factor, the remaining trajectory
        quantities are recomputed, and the scaled trajectory is processed again by the parent class.
        Otherwise, the trajectory is directly processed by the parent class.

        Args:
            traj (Trajectory): The trajectory to process.

        Returns:
            Trajectory: The processed trajectory.

        """
        if self.scaling != 1.0:
            # first pass: filter+extend+interpolate to get correct joint indices
            processed = super().process_trajectory(traj)

            traj_info = processed.info
            traj_data = processed.data
            free_jnt_pos_id = self.free_jnt_qpos_id[:, :3].reshape(-1)
            free_jnt_lin_vel_id = self.free_jnt_qvel_id[:, :3].reshape(-1)

            traj_data_scaled = TrajectoryData(
                qpos=traj_data.qpos.at[:, free_jnt_pos_id].mul(self.scaling),
                qvel=traj_data.qvel.at[:, free_jnt_lin_vel_id].mul(self.scaling),
                split_points=traj_data.split_points)
            traj_model = TrajectoryModel(njnt=traj_info.model.njnt, jnt_type=traj_info.model.jnt_type)
            traj_info_scaled = TrajectoryInfo(joint_names=traj_info.joint_names,
                                              model=traj_model, frequency=traj_info.frequency)
            scaled_traj = extend_motion(self.__class__.__name__, {},
                                        Trajectory(info=traj_info_scaled, data=traj_data_scaled))

            # second pass: reprocess with scaled traj, reset handler first
            self.th = None
            return super().process_trajectory(scaled_traj)

        return super().process_trajectory(traj)

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        Returns:
            Tuple[float, float]: Minimum and maximum healthy height of the root in meters, proportional to the
            scaling factor.

        """
        return (0.8*self.scaling, 1.1*self.scaling)

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml.

        Returns:
            str: Name of the upper body.

        """
        return "torso"

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Reduces the contacts of the model to speed up the simulation with Mjx.

        Mjx is slow in handling many complex contacts. Therefore, all contacts are disabled except the ones
        between the box feet and the floor.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        # --- disable all contacts in geom ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- define contacts between feet and floor --
        spec.add_pair(geomname1="floor", geomname2="foot_box_r")
        spec.add_pair(geomname1="floor", geomname2="foot_box_l")

        return spec

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Returns the list of sites for mimic.

        Returns:
            List[str]: List of site names.

        """
        return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
                "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
                "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
                "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml.

        Returns:
            str: Name of the root body.

        """
        return "pelvis"
