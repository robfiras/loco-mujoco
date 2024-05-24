import os
import warnings
from pathlib import Path

from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

import loco_mujoco
from loco_mujoco.environments import LocoEnv


class BaseHumanoid(LocoEnv):
    """
    Mujoco environment of a base humanoid model.

    """

    def __init__(self, use_muscles=False, use_box_feet=True, disable_arms=True, alpha_box_feet=0.5, **kwargs):
        """
        Constructor.

        Args:
            use_muscles (bool): If True, muscle actuators will be used, else torque actuators will be used.
            use_box_feet (bool): If True, boxes are used as feet (for simplification).
            disable_arms (bool): If True, all arm joints are removed and the respective
                actuators are removed from the action specification.
            alpha_box_feet (float): Alpha parameter of the boxes, which might be added as feet.

        """
        if use_muscles:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "humanoid" /
                        "humanoid_muscle.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "humanoid" /
                        "humanoid_torque.xml").as_posix()

        action_spec = self._get_action_specification(use_muscles)

        observation_spec = self._get_observation_specification()

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_muscles = use_muscles
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove, collision_groups = self._get_xml_modifications()

        xml_handle = mjcf.from_path(xml_path)
        if self._use_box_feet or self._disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                      motors_to_remove, equ_constr_to_remove)
            if self._use_box_feet:
                xml_handle = self._add_box_feet_to_xml_handle(xml_handle, alpha_box_feet)

            if self._disable_arms:
                xml_handle = self._reorient_arms(xml_handle)

        super().__init__(xml_handle, action_spec, observation_spec, collision_groups, **kwargs)

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset. Default is ["q_pelvis_tx", "q_pelvis_tz"].

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj).

        """

        if ignore_keys is None:
            ignore_keys = ["q_pelvis_tx", "q_pelvis_tz"]

        dataset = super().create_dataset(ignore_keys)

        return dataset

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
            if not self._use_muscles:
                motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot_box_r"]),
                                ("foot_l", ["foot_box_l"])]
        else:
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["r_foot"]),
                                ("front_foot_r", ["r_bofoot"]),
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

    def _has_fallen(self, obs, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.

        """

        pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])

        pelvis_height_condition = (obs[0] < -0.46) or (obs[0] > 0.1)
        pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 9)) or (pelvis_euler[2] > (np.pi / 9))

        pelvis_condition = (pelvis_height_condition or pelvis_tilt_condition
                            or pelvis_list_condition or pelvis_rotation_condition)

        lumbar_euler = self._get_from_obs(obs, ["q_lumbar_extension", "q_lumbar_bending", "q_lumbar_rotation"])

        lumbar_extension_condition = (lumbar_euler[0] < (-np.pi / 4)) or (lumbar_euler[0] > (np.pi / 10))
        lumbar_bending_condition = (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
        lumbar_rotation_condition = (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))

        lumbar_condition = (lumbar_extension_condition or lumbar_bending_condition or lumbar_rotation_condition)

        if return_err_msg:
            error_msg = ""
            if pelvis_height_condition:
                error_msg += "pelvis_height_condition violated.\n"
            if pelvis_tilt_condition:
                error_msg += "pelvis_tilt_condition violated.\n"
            if pelvis_list_condition:
                error_msg += "pelvis_list_condition violated.\n"
            if pelvis_rotation_condition:
                error_msg += "pelvis_rotation_condition violated.\n"
            if lumbar_extension_condition:
                error_msg += "lumbar_extension_condition violated.\n"
            if lumbar_bending_condition:
                error_msg += "lumbar_bending_condition violated.\n"
            if lumbar_rotation_condition:
                error_msg += "lumbar_rotation_condition violated.\n"

            return pelvis_condition or lumbar_condition, error_msg
        else:
            return (pelvis_condition or lumbar_condition)

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
    def generate(env, path, task="walk", dataset_type="real", debug=False, **kwargs):
        """
        Returns a Humanoid environment and a dataset corresponding to the specified task.

        Args:
            env (class): Humanoid class, either HumanoidTorque or HumanoidMuscle.
            path (str): Path to the dataset.
            task (str): Main task to solve. Either "walk" or "run".
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
            reference trajectory. This data does not perfectly match the kinematics
            and dynamics of this environment, hence it is more challenging. "perfect" uses
            a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.

        Returns:
            An MDP of a Torque Humanoid.

        """

        if "reward_type" in kwargs.keys():
            reward_type = kwargs["reward_type"]
            del kwargs["reward_type"]
        else:
            reward_type = "target_velocity"

        if task == "walk":
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)
            traj_path = Path(loco_mujoco.__file__).resolve().parent / path
            if "reward_params" in kwargs.keys():
                reward_params = kwargs["reward_params"]
                del kwargs["reward_params"]
            else:
                reward_params = dict(target_velocity=1.25)
        elif task == "run":
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)
            traj_path = Path(loco_mujoco.__file__).resolve().parent / path
            if "reward_params" in kwargs.keys():
                reward_params = kwargs["reward_params"]
                del kwargs["reward_params"]
            else:
                reward_params = dict(target_velocity=2.5)

        # Generate the MDP
        mdp = env(reward_type=reward_type, reward_params=reward_params, **kwargs)

        # Load the trajectory
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
            traj_params = dict(traj_path=traj_path,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq))
        elif dataset_type == "perfect":
            traj_data_freq = 100  # hz
            traj_files = mdp.load_dataset_and_get_traj_files(path, traj_data_freq)
            traj_params = dict(traj_files=traj_files,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq))

        mdp.load_trajectory(traj_params, warn=False)

        return mdp

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
    def _get_action_specification(use_muscles):
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """
        if use_muscles:
            action_spec = ["mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                           "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                           "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                           "mot_wrist_flex_l", "mot_wrist_dev_l", "glut_med1_r", "glut_med2_r",
                           "glut_med3_r", "glut_min1_r", "glut_min2_r", "glut_min3_r", "semimem_r", "semiten_r",
                           "bifemlh_r", "bifemsh_r", "sar_r", "add_long_r", "add_brev_r", "add_mag1_r", "add_mag2_r",
                           "add_mag3_r", "tfl_r", "pect_r", "grac_r", "glut_max1_r", "glut_max2_r", "glut_max3_r",
                           "iliacus_r", "psoas_r", "quad_fem_r", "gem_r", "peri_r", "rect_fem_r", "vas_med_r",
                           "vas_int_r", "vas_lat_r", "med_gas_r", "lat_gas_r", "soleus_r", "tib_post_r",
                           "flex_dig_r", "flex_hal_r", "tib_ant_r", "per_brev_r", "per_long_r", "per_tert_r",
                           "ext_dig_r", "ext_hal_r", "glut_med1_l", "glut_med2_l", "glut_med3_l", "glut_min1_l",
                           "glut_min2_l", "glut_min3_l", "semimem_l", "semiten_l", "bifemlh_l", "bifemsh_l",
                           "sar_l", "add_long_l", "add_brev_l", "add_mag1_l", "add_mag2_l", "add_mag3_l",
                           "tfl_l", "pect_l", "grac_l", "glut_max1_l", "glut_max2_l", "glut_max3_l",
                           "iliacus_l", "psoas_l", "quad_fem_l", "gem_l", "peri_l", "rect_fem_l",
                           "vas_med_l", "vas_int_l", "vas_lat_l", "med_gas_l", "lat_gas_l", "soleus_l",
                           "tib_post_l", "flex_dig_l", "flex_hal_l", "tib_ant_l", "per_brev_l", "per_long_l",
                           "per_tert_l", "ext_dig_l", "ext_hal_l", "ercspn_r", "ercspn_l", "intobl_r",
                           "intobl_l", "extobl_r", "extobl_l"]
        else:
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

        # find foot and attach box
        toe_l = xml_handle.find("body", "toes_l")
        size = np.array([0.112, 0.03, 0.05]) * scaling
        pos = np.array([-0.09, 0.019, 0.0]) * scaling
        toe_l.add("geom", name="foot_box_l", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_box_r", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "r_foot")
        bofoot_r = xml_handle.find("geom", "r_bofoot")
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
