import os.path
import warnings
from pathlib import Path
from copy import deepcopy
import numpy as np
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

import loco_mujoco
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import check_validity_task_mode_dataset


class Talos(LocoEnv):
    """
    Mujoco simulation of the Talos robot. Optionally, Talos can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "carry"],
                                     data_types=["real"])

    def __init__(self, disable_arms=False, disable_back_joint=False, hold_weight=False,
                 weight_mass=None, tmp_dir_name=None, **kwargs):
        """
        Constructor.

        """

        if hold_weight:
            assert disable_arms is True, "If you want Talos to carry a weight, please disable the arms. " \
                                         "They will be kept fixed."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "talos" / "talos.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["ground"]),
                            ("foot_r", ["right_foot"]),
                            ("foot_l", ["left_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint
        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]

        if disable_arms or hold_weight:
            xml_handle = mjcf.from_path(xml_path)

            if disable_arms or disable_back_joint:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)

            xml_path = []
            if hold_weight and weight_mass is not None:
                color_red = np.array([1.0, 0.0, 0.0, 1.0])
                xml_handle = self._add_weight(xml_handle, weight_mass, color_red)
                xml_path.append(self._save_xml_handle(xml_handle, tmp_dir_name))
            elif hold_weight and weight_mass is None:
                for i, w in enumerate(self._valid_weights):
                    color = self._get_box_color(i)
                    current_xml_handle = deepcopy(xml_handle)
                    current_xml_handle = self._add_weight(current_xml_handle, w, color)
                    xml_path.append(self._save_xml_handle(current_xml_handle, tmp_dir_name))
            else:
                xml_handle = self._reorient_arms(xml_handle)
                xml_path.append(self._save_xml_handle(xml_handle, tmp_dir_name))

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    def _get_grf_size(self):
        """
        Returns the size of the ground force vector.

        """

        return 6

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                              self._get_collision_force("floor", "foot_l")[:3]])

        return grf

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

    def get_mask(self, obs_to_hide):
        """
        This function returns a boolean mask to hide observations from a fully observable state.

        Args:
            obs_to_hide (tuple): A tuple of strings with names of objects to hide.
            Hidable objects are "positions", "velocities", "foot_forces", and "env_type".

        Returns:
            Mask in form of a np.array of booleans. True means that that the obs should be
            included, and False means that it should be discarded.

        """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)

        pos_dim, vel_dim = self._len_qpos_qvel()
        force_dim = self._get_grf_size()

        mask = []
        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(pos_dim, dtype=np.bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=np.bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=np.bool)]
            else:
                mask += [np.zeros(force_dim, dtype=np.bool)]
        else:
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."

        if self._hold_weight:
            if "weight" not in obs_to_hide:
                mask += [np.ones(1, dtype=np.bool)]
            else:
                mask += [np.zeros(1, dtype=np.bool)]
        else:
            assert "weight" not in obs_to_hide, "Creating a mask to hide the carried weight without activating " \
                                                "the latter is not allowed."

        return np.concatenate(mask).ravel()

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["l_arm_shz", "l_arm_shx", "l_arm_ely", "l_arm_elx", "l_arm_wry", "l_arm_wrx",
                                 "r_arm_shz", "r_arm_shx", "r_arm_ely", "r_arm_elx", "r_arm_wry", "r_arm_wrx"]
            motors_to_remove += ["l_arm_shz_actuator", "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator",
                                 "l_arm_wry_actuator", "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                                 "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator"]

        if self._disable_back_joint:
            joints_to_remove += ["back_bkz", "back_bky"]
            motors_to_remove += ["back_bkz_actuator", "back_bky_actuator"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        low, high = super(Talos, self)._get_observation_space()
        if self._hold_weight:
            low = np.concatenate([low, [self._valid_weights[0]]])
            high = np.concatenate([high, [self._valid_weights[-1]]])

        return low, high

    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            New observation vector (np.array).

        """

        obs = super(Talos, self)._create_observation(obs)
        if self._hold_weight:
            weight_mass = deepcopy(self._model.body("weight").mass)
            obs = np.concatenate([obs, weight_mass])

        return obs

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
        pelvis_y_condition = (obs[0] < -0.3) or (obs[0] > 0.1)
        pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
        pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
                            pelvis_list_condition or pelvis_rotation_condition)

        if not self._disable_back_joint:
            back_euler = self._get_from_obs(obs, ["q_back_bky", "q_back_bkz"])

            back_extension_condition = (back_euler[0] < (-np.pi / 4)) or (back_euler[0] > (np.pi / 10))
            back_rotation_condition = (back_euler[1] < -np.pi / 10) or (back_euler[1] > np.pi / 10)
            back_condition = (back_extension_condition or back_rotation_condition)
        else:
            back_condition = back_extension_condition = back_rotation_condition = False

        if return_err_msg:
            error_msg = ""
            if pelvis_y_condition:
                error_msg += "pelvis_y_condition violated.\n"
            elif pelvis_tilt_condition:
                error_msg += "pelvis_tilt_condition violated.\n"
            elif pelvis_list_condition:
                error_msg += "pelvis_list_condition violated.\n"
            elif pelvis_rotation_condition:
                error_msg += "pelvis_rotation_condition violated.\n"
            elif back_extension_condition:
                error_msg += "back_extension_condition violated.\n"
            elif back_rotation_condition:
                error_msg += "back_rotation_condition violated.\n"

            return pelvis_condition or back_condition, error_msg
        else:

            return pelvis_condition or back_condition

    def _get_box_color(self, ind):
        """
        Calculates the rgba color based on the index of the environment.

        Args:
            ind (int): Current index of the environment.

        Returns:
            rgba np.array.

        """

        red_rgba = np.array([1.0, 0.0, 0.0, 1.0])
        blue_rgba = np.array([0.2, 0.0, 1.0, 1.0])
        interpolation_var = ind / (len(self._valid_weights) - 1)
        color = blue_rgba + ((red_rgba - blue_rgba) * interpolation_var)

        return color

    @staticmethod
    def generate(task="walk", dataset_type="real", gamma=0.99, horizon=1000, random_env_reset=True, disable_arms=True,
                 disable_back_joint=False, use_foot_forces=False, random_start=True, init_step_no=None,
                 debug=False, hide_menu_on_startup=False):
        """
        Returns an Talos environment corresponding to the specified task.

        Args:
            task (str): Main task to solve. Either "walk" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.
            gamma (float): Discounting parameter of the environment.
            horizon (int): Horizon of the environment.
            random_env_reset (bool):  If True, a random environment is chosen after each episode. If False, it is
                sequentially iterated through the environment/model list.
            disable_arms (bool): If True, arms are disabled.
            disable_back_joint (bool): If True, the back joint is disabled.
            use_foot_forces (bool): If True, foot forces are added to the observation space.
            random_start (bool): If True, a random sample from the trajectories
                is chosen at the beginning of each time step and initializes the
                simulation according to that.
            init_step_no (int): If set, the respective sample from the trajectories
                is taken to initialize the simulation.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.
            hide_menu_on_startup (bool): If True, the menu overlay is hidden on startup.

        Returns:
            An MDP of the Talos Robot.

        """
        check_validity_task_mode_dataset(Talos.__name__, task, None, dataset_type,
                                         *Talos.valid_task_confs.get_all())

        reward_params = dict(target_velocity=1.25)

        # Generate the MDP
        if task == "walk":
            mdp = Talos(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                        disable_arms=disable_arms, disable_back_joint=disable_back_joint,
                        use_foot_forces=use_foot_forces, reward_type="target_velocity",
                        reward_params=reward_params, random_env_reset=random_env_reset,
                        hide_menu_on_startup=hide_menu_on_startup)
        elif task == "carry":
            mdp = Talos(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                        disable_arms=disable_arms, disable_back_joint=disable_back_joint,
                        use_foot_forces=use_foot_forces, hold_weight=True, reward_type="target_velocity",
                        reward_params=reward_params, random_env_reset=random_env_reset,
                        hide_menu_on_startup=hide_menu_on_startup)

        # Load the trajectory
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
            path = "datasets/humanoids/02-constspeed_TALOS.npz"
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent.parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(2, "mini_datasets")
                path = "/".join(path)

            traj_params = dict(traj_path=Path(loco_mujoco.__file__).resolve().parent.parent / path,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=True)

        elif dataset_type == "perfect":
            # todo: generate and add this dataset
            raise ValueError(f"currently not implemented.")

        mdp.load_trajectory(traj_params, warn=False)

        return mdp

    @staticmethod
    def _add_weight(xml_handle, mass, color):
        """
        Adds a weight to the Mujoco XML handle. The weight will
        be hold in front of Talos. Therefore, the arms will be
        reoriented.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find pelvis handle
        pelvis = xml_handle.find("body", "torso_1_link")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.25 0.1", pos="0.45 0 -0.20", group="0", rgba=color, mass=mass)

        # modify the arm orientation
        arm_right_4_link = xml_handle.find("body", "arm_right_4_link")
        arm_right_4_link.quat = [1.0,  0.0, -0.65, 0.0]
        arm_left_4_link = xml_handle.find("body", "arm_left_4_link")
        arm_left_4_link.quat = [1.0,  0.0, -0.65, 0.0]

        arm_right_6_link = xml_handle.find("body", "arm_right_6_link")
        arm_right_6_link.quat = [1.0,  0.0, -0.0, 1.0]
        arm_left_6_link = xml_handle.find("body", "arm_left_6_link")
        arm_left_6_link.quat = [1.0,  0.0, -0.0, 1.0]

        return xml_handle

    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorients the elbow to not collide with the hip.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # modify the arm orientation
        arm_right_4_link = xml_handle.find("body", "arm_right_4_link")
        arm_right_4_link.quat = [1.0,  0.0, -0.25, 0.0]
        arm_left_4_link = xml_handle.find("body", "arm_left_4_link")
        arm_left_4_link.quat = [1.0,  0.0, -0.25, 0.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_back_bkz", "back_bkz", ObservationType.JOINT_POS),
                            ("q_back_bky", "back_bky", ObservationType.JOINT_POS),
                            ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
                            ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
                            ("q_l_arm_ely", "l_arm_ely", ObservationType.JOINT_POS),
                            ("q_l_arm_elx", "l_arm_elx", ObservationType.JOINT_POS),
                            ("q_l_arm_wry", "l_arm_wry", ObservationType.JOINT_POS),
                            ("q_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_POS),
                            ("q_r_arm_shz", "r_arm_shz", ObservationType.JOINT_POS),
                            ("q_r_arm_shx", "r_arm_shx", ObservationType.JOINT_POS),
                            ("q_r_arm_ely", "r_arm_ely", ObservationType.JOINT_POS),
                            ("q_r_arm_elx", "r_arm_elx", ObservationType.JOINT_POS),
                            ("q_r_arm_wry", "r_arm_wry", ObservationType.JOINT_POS),
                            ("q_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_POS),
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_back_bkz", "back_bkz", ObservationType.JOINT_VEL),
                            ("dq_back_bky", "back_bky", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shz", "l_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shx", "l_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_ely", "l_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_l_arm_elx", "l_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wry", "l_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shz", "r_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shx", "r_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_ely", "r_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_r_arm_elx", "r_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wry", "r_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = ["back_bkz_actuator", "back_bky_actuator", "l_arm_shz_actuator",
                       "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator", "l_arm_wry_actuator",
                       "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                       "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator",
                       "hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator",
                       "knee_angle_r_actuator", "ankle_angle_r_actuator", "hip_flexion_l_actuator",
                       "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator",
                       "ankle_angle_l_actuator"]

        return action_spec
