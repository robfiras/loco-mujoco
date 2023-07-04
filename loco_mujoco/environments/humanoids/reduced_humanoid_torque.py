from pathlib import Path

from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments import BaseEnv


class ReducedHumanoidTorque(BaseEnv):
    """
    MuJoCo simulation of a simplified humanoid model with torque actuation.

    """

    def __init__(self, use_box_feet=False, disable_arms=False, tmp_dir_name=None, alpha_box_feet=0.5, **kwargs):
        """
        Constructor.

        Args:
            use_box_feet (bool): If True, boxes are used as feet (for simplification).
            disable_arms (bool): If True, all arm joints are removed and the respective
                actuators are removed from the action specification.
            tmp_dir_name (str): Specifies a name of a directory to which temporary files are
                written, if created. By default, temporary directory names are created automatically.
            alpha_box_feet (float): Alpha parameter of the boxes, which might be added as feet.

        """

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "reduced_humanoid_torque" /
                    "reduced_humanoid_torque.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove, collision_groups = self._get_xml_modifications()

        if self._use_box_feet or self._disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]
            xml_handle = mjcf.from_path(xml_path)

            xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                      motors_to_remove, equ_constr_to_remove)
            if self._use_box_feet:
                xml_handle = self._add_box_feet_to_xml_handle(xml_handle, alpha_box_feet)

            if self._disable_arms:
                xml_handle = self._reorient_arms(xml_handle)

            xml_path = self._save_xml_handle(xml_handle, tmp_dir_name)

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml. Also the required collision
        groups will be returned.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             names of equality constraints to remove, and names of collision groups to be used.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []
        if self._use_box_feet:
            joints_to_remove += ["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot_box_r"]),
                                ("foot_l", ["foot_box_l"])]
        else:
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot"]),
                                ("front_foot_r", ["bofoot"]),
                                ("foot_l", ["l_foot"]),
                                ("front_foot_l", ["l_bofoot"])]

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

        return joints_to_remove, motors_to_remove, equ_constr_to_remove, collision_groups

    def _has_fallen(self, obs):
        """
        Checks if a model has fallen. This has to be implemented for each environment.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the model has fallen for the current observation, False otherwise.

        """

        pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
        pelvis_condition = ((obs[0] < -0.46) or (obs[0] > 0.0)
                            or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                            or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                            or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10)))

        lumbar_euler = self._get_from_obs(obs, ["q_lumbar_extension", "q_lumbar_bending", "q_lumbar_rotation"])
        lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                            or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                            or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5)))

        return pelvis_condition or lumbar_condition

    def _setup_ground_force_statistics(self):
        """
        Returns a running average method for the mean ground forces.

        """

        grf_vec_size = self._get_grf_size()
        mean_grf = RunningAveragedWindow(shape=(grf_vec_size,), window_size=self._n_substeps)

        return mean_grf

    def _get_grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        if self._use_box_feet:
            return 6
        else:
            return 12

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        if self._use_box_feet:
            grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                                  self._get_collision_force("floor", "foot_l")[:3]])
        else:
            grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                                  self._get_collision_force("floor", "front_foot_r")[:3],
                                  self._get_collision_force("floor", "foot_l")[:3],
                                  self._get_collision_force("floor", "front_foot_l")[:3]])

        return grf

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [  # ------------- JOINT POS -------------
            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
            # --- lower limb right ---
            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
            ("q_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
            ("q_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
            # --- lower limb left ---
            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
            ("q_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
            ("q_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
            # --- lumbar ---
            ("q_lumbar_extension", "lumbar_extension", ObservationType.JOINT_POS),
            ("q_lumbar_bending", "lumbar_bending", ObservationType.JOINT_POS),
            ("q_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_POS),
            # --- upper body right ---
            ("q_arm_flex_r", "arm_flex_r", ObservationType.JOINT_POS),
            ("q_arm_add_r", "arm_add_r", ObservationType.JOINT_POS),
            ("q_arm_rot_r", "arm_rot_r", ObservationType.JOINT_POS),
            ("q_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_POS),
            ("q_pro_sup_r", "pro_sup_r", ObservationType.JOINT_POS),
            ("q_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_POS),
            ("q_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_POS),
            # --- upper body left ---
            ("q_arm_flex_l", "arm_flex_l", ObservationType.JOINT_POS),
            ("q_arm_add_l", "arm_add_l", ObservationType.JOINT_POS),
            ("q_arm_rot_l", "arm_rot_l", ObservationType.JOINT_POS),
            ("q_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_POS),
            ("q_pro_sup_l", "pro_sup_l", ObservationType.JOINT_POS),
            ("q_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_POS),
            ("q_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_POS),

            # ------------- JOINT VEL -------------
            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
            # --- lower limb right ---
            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
            ("dq_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_VEL),
            ("dq_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_VEL),
            # --- lower limb left ---
            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL),
            ("dq_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_VEL),
            ("dq_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_VEL),
            # --- lumbar ---
            ("dq_lumbar_extension", "lumbar_extension", ObservationType.JOINT_VEL),
            ("dq_lumbar_bending", "lumbar_bending", ObservationType.JOINT_VEL),
            ("dq_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_VEL),
            # --- upper body right ---
            ("dq_arm_flex_r", "arm_flex_r", ObservationType.JOINT_VEL),
            ("dq_arm_add_r", "arm_add_r", ObservationType.JOINT_VEL),
            ("dq_arm_rot_r", "arm_rot_r", ObservationType.JOINT_VEL),
            ("dq_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_VEL),
            ("dq_pro_sup_r", "pro_sup_r", ObservationType.JOINT_VEL),
            ("dq_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_VEL),
            ("dq_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_VEL),
            # --- upper body left ---
            ("dq_arm_flex_l", "arm_flex_l", ObservationType.JOINT_VEL),
            ("dq_arm_add_l", "arm_add_l", ObservationType.JOINT_VEL),
            ("dq_arm_rot_l", "arm_rot_l", ObservationType.JOINT_VEL),
            ("dq_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_VEL),
            ("dq_pro_sup_l", "pro_sup_l", ObservationType.JOINT_VEL),
            ("dq_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_VEL),
            ("dq_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = ["mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
                       "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
                       "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
                       "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
                       "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
                       "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
                       "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
                       "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        return action_spec

    @staticmethod
    def _add_box_feet_to_xml_handle(xml_handle, alpha_box_feet, scaling=1.0):
        """
        Adds box feet to Mujoco XML handle and makes old feet non-collidable.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find foot and attach bricks
        toe_l = xml_handle.find("body", "toes_l")
        size = np.array([0.112, 0.03, 0.05]) * scaling
        pos = np.array([-0.09, 0.019, 0.0]) * scaling
        toe_l.add("geom", name="foot_brick_l", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_brick_r", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "foot")
        bofoot_r = xml_handle.find("geom", "bofoot")
        foot_l = xml_handle.find("geom", "l_foot")
        bofoot_l = xml_handle.find("geom", "l_bofoot")
        foot_r.contype = 0
        foot_r.conaffinity = 0
        bofoot_r.contype = 0
        bofoot_r.conaffinity = 0
        foot_l.contype = 0
        foot_l.conaffinity = 0
        bofoot_l.contype = 0
        bofoot_l.conaffinity = 0

        return xml_handle

    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorients the arm of a humanoid model given its Mujoco XML handle.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        h = xml_handle.find("body", "humerus_l")
        h.quat = [1.0, -0.1, -1.0, -0.1]
        h = xml_handle.find("body", "ulna_l")
        h.quat = [1.0, 0.6, 0.0, 0.0]
        h = xml_handle.find("body", "humerus_r")
        h.quat = [1.0, 0.1, 1.0, -0.1]
        h = xml_handle.find("body", "ulna_r")
        h.quat = [1.0, -0.6, 0.0, 0.0]

        return xml_handle
