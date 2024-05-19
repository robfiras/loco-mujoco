import os
import warnings
from pathlib import Path
from copy import deepcopy

from dm_control import mjcf

from mushroom_rl.utils.running_stats import *

import loco_mujoco
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.utils.reward import MultiTargetVelocityReward
from loco_mujoco.utils import check_validity_task_mode_dataset


class BaseHumanoid4Ages(BaseHumanoid):
    """
    Mujoco environment of 4 simplified humanoid models.
    At the beginning of each episode, one of the four humanoid models are
    sampled and used to simulate a trajectory. The different humanoids should
    resemble an adult, a teenager (∼12 years), a child (∼5 years), and a
    toddler (∼1-2 years). This environment can be partially observable by
    using state masks to hide the humanoid type indicator from the policy.

    """

    def __init__(self, scaling=None, scaling_trajectory_map=None, use_muscles=False,
                 use_box_feet=True, disable_arms=True, alpha_box_feet=0.5, **kwargs):
        """
        Constructor.

        Args:
            scaling (float or list): Scaling of the humanoids. Should be in > 0.0.
            scaling_trajectory_map (list): A list that contains tuples of two integers
                for each scaling. Given a set of trajectories, they define the range of
                the valid trajectory numbers for each scaling factor.
            use_muscles (bool): If True, muscle actuators are used, else one torque actuator
                ss used per joint.
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

        self._use_muscles = use_muscles
        action_spec = self._get_action_specification(use_muscles)

        observation_spec = self._get_observation_specification()

        self._hidable_obs = ("positions", "velocities", "foot_forces", "env_type")

        # 0.4 ~ 1-2 year old infant who just started walking
        # 0.6 ~ 5 year old boy
        # 0.8 ~ 12 year old boy
        # 1.0 ~ 20 year old man
        self._default_scalings = [0.4, 0.6, 0.8, 1.0]
        if scaling is None:
            self._scalings = self._default_scalings
        else:
            if type(scaling) == list:
                self._scalings = scaling
            else:
                self._scalings = [scaling]

        self._scaling_trajectory_map = scaling_trajectory_map

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_box_feet = use_box_feet
        self._disable_arms = disable_arms
        joints_to_remove, motors_to_remove, equ_constr_to_remove, collision_groups = self._get_xml_modifications()

        xml_handle = mjcf.from_path(xml_path)
        xml_handles = [self.scale_body(deepcopy(xml_handle), scaling, use_muscles) for scaling in self._scalings]

        if use_box_feet or disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            for handle, scale in zip(xml_handles, self._scalings):
                handle = self._delete_from_xml_handle(handle, joints_to_remove,
                                                      motors_to_remove, equ_constr_to_remove)

                if use_box_feet:
                    handle = self._add_box_feet_to_xml_handle(handle, alpha_box_feet, scale)

                if disable_arms:
                    self._reorient_arms(handle)

        # call gran-parent
        super(BaseHumanoid, self).__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

        if scaling_trajectory_map is not None and self.trajectories is None:
            warnings.warn("You have defined a scaling_trajectory_map, but no trajectory was defined. The former "
                          "will have no effect.")

    def setup(self, obs):
        """
        Function to setup the initial state of the simulation. Initialization can be done either
        randomly, from a certain initial, or from the default initial state of the model. If random
        is chosen, a trajectory is sampled based on the current model.

        Args:
            obs (np.array): Observation to initialize the environment from;

        """

        self._reward_function.reset_state()

        if obs is not None:
            raise TypeError("Initializing the environment from an observation is "
                            "not allowed in this environment.")
        else:
            if not self.trajectories and self._random_start:
                raise ValueError("Random start not possible without trajectory data.")
            elif not self.trajectories and self._init_step_no is not None:
                raise ValueError("Setting an initial step is not possible without trajectory data.")
            elif self._init_step_no is not None and self._random_start:
                raise ValueError("Either use a random start or set an initial step, not both.")

            if self.trajectories is not None:
                if self._random_start:
                    if self._scaling_trajectory_map:
                        curr_model = self._current_model_idx
                        valid_traj_range = self._scaling_trajectory_map[curr_model]
                        traj_no = np.random.randint(valid_traj_range[0], valid_traj_range[1])
                        sample = self.trajectories.reset_trajectory(traj_no=traj_no)
                    else:
                        sample = self.trajectories.reset_trajectory()
                elif self._init_step_no:
                    traj_len = self.trajectories.trajectory_length
                    n_traj = self.trajectories.nnumber_of_trajectories
                    assert self._init_step_no <= traj_len * n_traj
                    substep_no = int(self._init_step_no % traj_len)
                    traj_no = int(self._init_step_no / traj_len)
                    sample = self.trajectories.reset_trajectory(substep_no, traj_no)
                self.set_sim_state(sample)

    def load_trajectory(self, traj_params, scaling_trajectory_map=None, warn=True):
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj_params (dict): Dictionary of parameters needed to load trajectories;
            scaling_trajectory_map (list): A list that contains tuples of two integers
            for each scaling. Given a set of trajectories, they define the range of
            the valid trajectory numbers for each scaling factor.
            warn (bool): If True, a warning will be raised if scaling_trajectory_map is not set
            or the trajectory ranges are violated.

        """

        super().load_trajectory(traj_params, warn)

        if scaling_trajectory_map is None:
            if self._scaling_trajectory_map is None and type(self._scalings) is list and len(self._scalings) > 1:
                if warn:
                    warnings.warn("\"scaling_trajectory_map\" is not defined! Loading the default map, which assumes that "
                                  "the trajectory contains an equal number of trajectories for all scalings and that"
                                  "they are ordered in the following order %s." % self._scalings)
                n_trajs_per_scaling = self.trajectories.number_of_trajectories / len(self._scalings)
                assert n_trajs_per_scaling.is_integer(), "Failed to construct the default" \
                                                         "\"scaling_trajectory_map\". The number of trajectory " \
                                                         "can not be divided by the number of scalings!"
                n_trajs_per_scaling = int(n_trajs_per_scaling)
                current_low_idx = 0
                self._scaling_trajectory_map = []
                for i in range(self.trajectories.number_of_trajectories):
                    current_high_idx = current_low_idx + n_trajs_per_scaling
                    self._scaling_trajectory_map.append((current_low_idx, current_high_idx))
                    current_low_idx = current_high_idx
            else:
                # only one scaling used
                self._scaling_trajectory_map = None
        else:
            self._scaling_trajectory_map = scaling_trajectory_map

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

        pos_dim = len(self._get_joint_pos()) - 2
        vel_dim = len(self._get_joint_vel())
        force_dim = self._get_grf_size()
        env_id_dim = len(self._get_env_id_map(0, self.n_all_models))

        mask = []

        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim, dtype=bool)]
        else:
            mask += [np.zeros(pos_dim, dtype=bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=bool)]
            else:
                mask += [np.zeros(force_dim, dtype=bool)]
        else:
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."
        if self.more_than_one_env:
            if "env_type" not in obs_to_hide:
                mask += [np.ones(env_id_dim, dtype=bool)]
            else:
                mask += [np.zeros(env_id_dim, dtype=bool)]
        else:
            assert "env_type" not in obs_to_hide, "Creating a mask to hide the env type without having more than " \
                                                  "one env is not allowed."

        return np.concatenate(mask).ravel()

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        low, high = super(BaseHumanoid4Ages, self)._get_observation_space()

        len_env_map = len(self._get_env_id_map(self._current_model_idx, self.n_all_models))
        low = np.concatenate([low, np.zeros(len_env_map)])
        high = np.concatenate([high, np.ones(len_env_map)])
        return low, high

    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;

        Returns:
            New observation vector (np.array);

        """

        obs = super(BaseHumanoid4Ages, self)._create_observation(obs)

        if self.more_than_one_env:
            model_idx = self._current_model_idx
        else:
            model_idx = self._default_scalings.index(self._scalings[0])

        env_id_map = self._get_env_id_map(model_idx, self.n_all_models)
        obs = np.concatenate([obs, env_id_map])
        return obs

    def _get_reward_function(self, reward_type, reward_params):
        """
        Constructs a reward function.

        Args:
            reward_type (string): Name of the reward.
            reward_params (dict): Parameters of the reward function.

        Returns:
            Reward function.

        """

        if reward_type == "multi_target_velocity":
            x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
            assert len(x_vel_idx) == 1
            x_vel_idx = x_vel_idx[0]
            env_id_len = len(self._get_env_id_map(0, self.n_all_models))
            goal_reward_func = MultiTargetVelocityReward(x_vel_idx=x_vel_idx, scalings=self._default_scalings,
                                                         env_id_len=env_id_len, **reward_params)
        else:
             goal_reward_func = super()._get_reward_function(reward_type, reward_params)

        return goal_reward_func

    @staticmethod
    def scale_body(xml_handle, scaling, use_muscles):
        """
        This function scales the kinematics and dynamics of the humanoid model given a Mujoco XML handle.

        Args:
            xml_handle: Handle to Mujoco XML.
            scaling (float): Scaling factor.

        Returns:
            Modified Mujoco XML handle.

        """

        body_scaling = scaling
        mesh_handle = xml_handle.find_all("mesh")

        head_geoms = ["hat_skull", "hat_jaw", "hat_ribs_cap"]

        for h in mesh_handle:
            if h.name not in head_geoms:  # don't scale head
                h.scale *= body_scaling

        for h in xml_handle.find_all("geom"):
            if h.name in head_geoms:  # change position of head
                h.pos = [0.0, -0.5 * (1 - scaling), 0.0]

        body_handle = xml_handle.find_all("body")
        for h in body_handle:
            h.pos *= body_scaling
            h.inertial.mass *= body_scaling ** 3
            # Diagonal elements of the inertia matrix change v with scaling.
            # As all off-diagonal elements are 0 here.
            h.inertial.fullinertia *= body_scaling ** 5
            assert np.array_equal(h.inertial.fullinertia[3:], np.zeros(3)), "Some of the diagonal elements of the" \
                                                                            "inertia matrix are not zero! Scaling is" \
                                                                            "not done correctly. Double-Check!"

        if use_muscles:
            for h in body_handle:
                for s in h.site:
                    s.pos *= body_scaling

            actuator_handle = xml_handle.find_all("actuator")
            for h in actuator_handle:
                if "mot" not in h.name:
                    h.force *= body_scaling ** 2

        if not use_muscles:
            actuator_handle = xml_handle.find_all("actuator")
            for h in actuator_handle:
                h.gear *= body_scaling ** 2

        return xml_handle

    @staticmethod
    def generate(env, path, task="walk", mode="all", dataset_type="real", n_models=None, debug=False, **kwargs):
        """
        Returns a Humanoid environment corresponding to the specified task.

        Args:
            env (class): Humanoid class, either HumanoidTorque4Ages or HumanoidMuscle4Ages.
            path (str): Path to the dataset.
            task (str): Main task to solve. Either "walk" or "run".
            mode (str): Mode of the environment. Either "all" (sample between all humanoid envs), "1"
            (smallest humanoid), "2" (second smallest humanoid), "3" (teenage humanoid), and "4" (adult humanoid).
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
            reference trajectory. This data does not perfectly match the kinematics
            and dynamics of this environment, hence it is more challenging. "perfect" uses
            a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.

        Returns:
            An MDP of a set of Torque or Muscle Humanoid of different sizes.

        """

        if mode == "all":
            dataset_suffix = "_all.npz"
            scaling = None
        elif mode == "1":
            dataset_suffix = "_1.npz"
            scaling = 0.4
        elif mode == "2":
            dataset_suffix = "_2.npz"
            scaling = 0.6
        elif mode == "3":
            dataset_suffix = "_3.npz"
            scaling = 0.8
        elif mode == "4":
            dataset_suffix = "_4.npz"
            scaling = 1.0

        if dataset_type == "real":
            local_path = path + dataset_suffix
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / local_path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                local_path = local_path.split("/")
                local_path.insert(3, "mini_datasets")
                local_path = "/".join(local_path)
            traj_path = Path(loco_mujoco.__file__).resolve().parent / local_path
        elif dataset_type == "perfect":
            local_path = path + dataset_suffix
            traj_path = Path(loco_mujoco.__file__).resolve().parent / local_path

        if task == "walk":
            reward_params = dict(target_velocity=1.25)
        elif task == "run":
            reward_params = dict(target_velocity=2.5)

        # Generate the MDP
        mdp = env(scaling=scaling, reward_type="multi_target_velocity", reward_params=reward_params, **kwargs)

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
            traj_files = mdp.load_dataset_and_get_traj_files(traj_path, traj_data_freq)
            traj_params = dict(traj_files=traj_files,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq))

        mdp.load_trajectory(traj_params, warn=False)

        return mdp

    @property
    def n_all_models(self):
        return len(self._default_scalings)
