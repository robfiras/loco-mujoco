import os
import warnings
from pathlib import Path
from copy import deepcopy

from dm_control import mjcf
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.angles import mat_to_euler, euler_to_mat

import loco_mujoco
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils.reward import VelocityVectorReward
from loco_mujoco.utils.math import rotate_obs
from loco_mujoco.utils.goals import GoalDirectionVelocity
from loco_mujoco.utils.math import mat2angle_xy, angle2mat_xy, transform_angle_2pi
from loco_mujoco.utils.checks import check_validity_task_mode_dataset


class UnitreeA1(LocoEnv):

    """
    Description
    ------------

    Mujoco environment of Unitree A1 model.

    Tasks
    -----------------
    * **Simple**: The robot has to walk forward with a fixed speed of 0.5 m/s.
    * **Hard**: The robot has to walk in 8 different directions with a fixed speed of 0.5 m/s.


    Dataset Types
    -----------------
    The available dataset types for this environment can be found at: :ref:`env-label`.


    Observation Space
    -----------------

    The observation space has the following properties *by default* (i.e., only obs with Disabled == False):

    | For simple task: :code:`(min=-inf, max=inf, dim=37, dtype=float32)`
    | For hard task: :code:`(min=-inf, max=inf, dim=37, dtype=float32)`

    Some observations are **disabled by default**, but can be turned on. The detailed observation space is:

    ===== ========================================================= ========= ========= ======== === ========================
    0     Position of Joint trunk_tz                                -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    1     Position of Joint trunk_list                              -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    2     Position of Joint trunk_tilt                              -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    3     Position of Joint trunk_rotation                          -inf      inf       False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    4     Position of Joint FR_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    5     Position of Joint FR_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    6     Position of Joint FR_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    7     Position of Joint FL_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    8     Position of Joint FL_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    9     Position of Joint FL_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    10    Position of Joint RR_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    11    Position of Joint RR_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    12    Position of Joint RR_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    13    Position of Joint RL_hip_joint                            -0.802851 0.802851  False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    14    Position of Joint RL_thigh_joint                          -1.0472   4.18879   False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    15    Position of Joint RL_calf_joint                           -2.69653  -0.916298 False    1   Angle [rad]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    16    Velocity of Joint trunk_tx                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    17    Velocity of Joint trunk_ty                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    18    Velocity of Joint trunk_tz                                -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    19    Velocity of Joint trunk_list                              -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    20    Velocity of Joint trunk_tilt                              -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    21    Velocity of Joint trunk_rotation                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    22    Velocity of Joint FR_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    23    Velocity of Joint FR_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    24    Velocity of Joint FR_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    25    Velocity of Joint FL_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    26    Velocity of Joint FL_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    27    Velocity of Joint FL_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    28    Velocity of Joint RR_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    29    Velocity of Joint RR_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    30    Velocity of Joint RR_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    31    Velocity of Joint RL_hip_joint                            -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    32    Velocity of Joint RL_thigh_joint                          -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    33    Velocity of Joint RL_calf_joint                           -inf      inf       False    1   Angular Velocity [rad/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    34    Desired Velocity Angle represented as Sine-Cosine Feature 0.0       1         False    2   None
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    36    Desired Velocity                                          0.0       inf       False    1   Velocity [m/s]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    37    3D linear Forces between Front Left Foot and Floor        0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    40    3D linear Forces between Front Right Foot and Floor       0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    43    3D linear Forces between Back Left Foot and Floor         0.0       inf       True     3   Force [N]
    ----- --------------------------------------------------------- --------- --------- -------- --- ------------------------
    46    3D linear Forces between Back Right Foot and Floor        0.0       inf       True     3   Force [N]
    ===== ========================================================= ========= ========= ======== === ========================

    Action Space
    ------------

    | The action space has the following properties *by default* (i.e., only actions with Disabled == False):
    | :code:`(min=-1, max=1, dim=12, dtype=float32)`

    ===== =========== =========== =========== ========
    Index Name in XML Control Min Control Max Disabled
    ===== =========== =========== =========== ========
    0     FR_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    1     FR_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    2     FR_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    3     FL_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    4     FL_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    5     FL_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    6     RR_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    7     RR_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    8     RR_calf     -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    9     RL_hip      -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    10    RL_thigh    -1.0        1.0         False
    ----- ----------- ----------- ----------- --------
    11    RL_calf     -1.0        1.0         False
    ===== =========== =========== =========== ========


    Rewards
    --------

    Reward function based on the difference between the desired velocity vector and the actual center of mass velocity
    vector in horizontal plane. The desired velocity vector is given by the dataset to imitate.

    **Class**: :class:`loco_mujoco.utils.reward.VelocityVectorReward`


    Initial States
    ---------------

    The initial state is sampled by default from the dataset to imitate.

    Terminal States
    ----------------

    The terminal state is reached when the robot falls, or rather starts falling. The condition to check if the robot
    is falling is based on the orientation of the robot and the height of the center of mass. More details can be found
    in the  :code:`_has_fallen` method of the environment.

    Methods
    ------------

    """

    valid_task_confs = ValidTaskConf(tasks=["simple", "hard"],
                                     data_types=["real", "perfect"])

    def __init__(self, action_mode="torque", setup_random_rot=False,
                 default_target_velocity=0.5, camera_params=None, **kwargs):
        """
        Constructor.

        Args:
            action_mode (str): Either "torque", "position", or "position_difference". Defines the action controller.
            setup_random_rot (bool): If True, the robot is initialized with a random rotation.
            default_target_velocity (float): Default target velocity set in the goal, when no trajectory
                data is provided.
            camera_params (dict): Dictionary defining some of the camera parameters for visualization.

        """

        # Choose xml file (either for torque or position control)
        if action_mode == "torque":
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_torque.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_position.xml").as_posix()

        self._action_mode = action_mode
        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("foot_FR", ["FR_foot"]),
                            ("foot_FL", ["FL_foot"]),
                            ("foot_RR", ["RR_foot"]),
                            ("foot_RL", ["RL_foot"])]

        # append observation_spec with the direction arrow and add it to the xml file
        observation_spec.append(("dir_arrow", "dir_arrow", ObservationType.SITE_ROT))
        xml_handle = self._add_dir_vector_to_xml_handle(mjcf.from_path(xml_path))

        self.setup_random_rot = setup_random_rot

        # setup goal including the desired direction and velocity
        self._goal = GoalDirectionVelocity()
        self._goal.set_goal(0.0, default_target_velocity)

        if camera_params is None:
            # make the camera by default a bit higher
            camera_params = dict(follow=dict(distance=3.5, elevation=-20.0, azimuth=90.0))
        super().__init__(xml_handle, action_spec, observation_spec,  collision_groups,
                         camera_params=camera_params, **kwargs)

    def setup(self, obs):
        """
        Function to setup the initial state of the simulation. Initialization can be done either
        randomly, from a certain initial, or from the default initial state of the model.

        Args:
            obs (np.array): Observation to initialize the environment from;

        """

        self._reward_function.reset_state()

        if obs is not None:
            self._init_sim_from_obs(obs)
        else:
            if not self.trajectories and self._random_start:
                raise ValueError("Random start not possible without trajectory data.")
            elif not self.trajectories and self._init_step_no is not None:
                raise ValueError("Setting an initial step is not possible without trajectory data.")
            elif self._init_step_no is not None and self._random_start:
                raise ValueError("Either use a random start or set an initial step, not both.")

            if self.trajectories is not None:
                if self._random_start:
                    sample = self.trajectories.reset_trajectory()
                    if self.setup_random_rot:
                        angle = np.random.uniform(0, 2 * np.pi)
                        sample = rotate_obs(sample, angle,  *self._get_relevant_idx_rotation())
                elif self._init_step_no:
                    traj_len = self.trajectories.trajectory_length
                    n_traj = self.trajectories.nnumber_of_trajectories
                    assert self._init_step_no <= traj_len * n_traj
                    substep_no = int(self._init_step_no % traj_len)
                    traj_no = int(self._init_step_no / traj_len)
                    sample = self.trajectories.reset_trajectory(substep_no, traj_no)
                else:
                    # sample random trajectory and use the first sample
                    sample = self.trajectories.reset_trajectory(substep_no=0)
                    if self.setup_random_rot:
                        angle = np.random.uniform(0, 2 * np.pi)
                        sample = rotate_obs(sample, angle,  *self._get_relevant_idx_rotation())

                # set the goal
                rot_mat = self.trajectories.get_from_sample(sample, "dir_arrow")
                angle = mat2angle_xy(rot_mat)
                desired_vel = self.trajectories.get_from_sample(sample, "goal_speed")
                self._goal.set_goal(angle, desired_vel)

                # set the state of the simulation
                self.set_sim_state(sample)

    def set_sim_state(self, sample):
        """
        Sets the state of the simulation according to an observation.

        Args:
            sample (list or np.array): Sample used to set the state of the simulation.

        """

        # set simulation state
        sample = sample[:-1]    # remove goal velocity
        super(UnitreeA1, self).set_sim_state(sample)

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj).

        """

        if self._dataset is None:

            if ignore_keys is None:
                ignore_keys = ["q_trunk_tx", "q_trunk_ty"]

            if self.trajectories is not None:
                rot_mat_idx_arrow = self._get_idx("dir_arrow")
                state_callback_params = dict(rot_mat_idx_arrow=rot_mat_idx_arrow,
                                             goal_velocity_idx=self._goal_velocity_idx)
                dataset = self.trajectories.create_dataset(ignore_keys=ignore_keys,
                                                           state_callback=self._modify_observation_callback,
                                                           state_callback_params=state_callback_params)
            else:
                raise ValueError("No trajectory was passed to the environment. "
                                 "To create a dataset pass a trajectory first.")

            self._dataset = deepcopy(dataset)

            return dataset

        else:
            return deepcopy(self._dataset)

    def get_kinematic_obs_mask(self):
        """
        Returns a mask (np.array) for the observation specified in observation_spec (or part of it).

        """

        return np.arange(len(self.obs_helper.observation_spec))

    def load_dataset_and_get_traj_files(self, dataset_path, freq=None):
        """
        Calculates a dictionary containing the kinematics given a dataset. If freq is provided,
        the x and z positions are calculated based on the velocity.

        Args:
            dataset_path (str): Path to the dataset.
            freq (float): Frequency of the data in obs.

        Returns:
            Dictionary containing the keys specified in observation_spec with the corresponding
            values from the dataset.

        """

        dataset = np.load(str(Path(loco_mujoco.__file__).resolve().parent / dataset_path))
        self._dataset = deepcopy({k: d for k, d in dataset.items()})

        states = dataset["states"]
        last = dataset["last"]

        states = np.atleast_2d(states)
        rel_keys = [obs_spec[0] for obs_spec in self.obs_helper.observation_spec]
        num_data = len(states)
        trajectories = dict()
        for i, key in enumerate(rel_keys):
            if i < 2:
                if freq is None:
                    # fill with zeros for x and y position
                    data = np.zeros(num_data)
                else:
                    # compute positions from velocities
                    dt = 1 / float(freq)
                    assert len(states) > 2
                    vel_idx = rel_keys.index("d" + key) - 2
                    data = [0.0]
                    for j, o in enumerate(states[:-1, vel_idx], 1):
                        if last is not None and last[j - 1] == 1:
                            data.append(0.0)
                        else:
                            data.append(data[-1] + dt * o)
                    data = np.array(data)
            elif key == "dir_arrow":
                sin_cos = states[:, i-2:i]
                angle = np.arctan2(sin_cos[:, 1], sin_cos[:, 0]) #+ np.pi/2
                if num_data > 1:
                    data = [angle2mat_xy(a).reshape((9,)) for a in angle]
                else:
                    data = angle2mat_xy(angle).reshape((9,))
                # calculate goal_speed
                dq_trunk_tx = states[:, rel_keys.index("dq_trunk_tx")-2]
                dq_trunk_ty = states[:, rel_keys.index("dq_trunk_ty")-2]
                vels = np.stack([dq_trunk_tx, dq_trunk_ty], axis=1)
                goal_speed = np.linalg.norm(vels, axis=1)
                goal_speed = np.mean(goal_speed) * np.ones_like(goal_speed)
                trajectories["goal_speed"] = goal_speed
            else:
                data = states[:, i - 2]
            trajectories[key] = data

        # add split points
        if len(states) > 2:
            trajectories["split_points"] = np.concatenate([[0], np.squeeze(np.argwhere(last == 1) + 1)])

        return trajectories

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """
        dir_arrow_idx = self._get_idx("dir_arrow")
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])
        sin_cos_angle_low = [-1, -1]
        sin_cos_angle_high = [1, 1]
        goal_vel_low = -np.inf
        goal_vel_high = np.inf
        sim_low = np.concatenate([sim_low[:dir_arrow_idx[0]], sin_cos_angle_low, [goal_vel_low]])
        sim_high = np.concatenate([sim_high[:dir_arrow_idx[0]], sin_cos_angle_high, [goal_vel_high]])

        if self._use_foot_forces:
            grf_low, grf_high = (-np.ones((12,)) * np.inf,
                                 np.ones((12,)) * np.inf)
            return (np.concatenate([sim_low, grf_low]),
                    np.concatenate([sim_high, grf_high]))
        else:
            return sim_low, sim_high

    def _simulation_post_step(self):
        """
        Sets the correct rotation of the goal arrow and the calculates the
        statistics of the ground forces if required. This function is
        called after each step.

        """

        self._set_goal_arrow()
        super()._simulation_post_step()

    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;

        Returns:
            New observation vector (np.array);

        """
        rot_mat_idx_arrow = self._get_idx("dir_arrow")

        obs = np.concatenate([obs[2:], [self._goal.get_velocity()]]).flatten()

        obs = self._modify_observation_callback(obs, rot_mat_idx_arrow, self._goal_velocity_idx)

        if self._use_foot_forces:
            obs = np.concatenate([obs,
                                  self.mean_grf.mean / 1000.,
                                  ]).flatten()

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

        if reward_type == "velocity_vector":
            x_vel_idx = self.get_obs_idx("dq_trunk_tx")[0]
            y_vel_idx = self.get_obs_idx("dq_trunk_ty")[0]
            angle_idx = [-3, -2]
            goal_vel_idx = [-1]
            goal_reward_func = VelocityVectorReward(x_vel_idx=x_vel_idx, y_vel_idx=y_vel_idx,
                                                    angle_idx=angle_idx, goal_vel_idx=goal_vel_idx)
        else:
            goal_reward_func = super()._get_reward_function(reward_type, reward_params)

        return goal_reward_func

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

        trunk_euler = self._get_from_obs(obs, ["q_trunk_list", "q_trunk_tilt"])
        trunk_height = self._get_from_obs(obs, ["q_trunk_tz"])

        trunk_list_condition = (trunk_euler[0] < -0.2793) or (trunk_euler[0] > 0.2793)
        trunk_tilt_condition = (trunk_euler[1] < -0.192) or (trunk_euler[1] > 0.192)
        trunk_height_condition = trunk_height[0] < -.24
        trunk_condition = (trunk_list_condition or trunk_tilt_condition or trunk_height_condition)

        if return_err_msg:
            error_msg = ""
            if trunk_list_condition:
                error_msg += "trunk_list_condition violated.\n"
            elif trunk_tilt_condition:
                error_msg += "trunk_tilt_condition violated.\n"
            elif trunk_height_condition:
                error_msg += "trunk_height_condition violated. %f \n" % trunk_height

            return trunk_condition, error_msg
        else:
            return trunk_condition

    def _get_relevant_idx_rotation(self):
        """
        Returns the indices relevant for rotating the observation space
        around the vertical axis.

        """

        keys = self.obs_helper.get_all_observation_keys()
        idx_rot = keys.index("q_trunk_rotation")
        idx_xvel = keys.index("dq_trunk_tx")
        idx_yvel = keys.index("dq_trunk_ty")
        return idx_rot, idx_xvel, idx_yvel

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array).

        """

        grf = np.concatenate([self._get_collision_force("floor", "foot_FL")[:3],
                              self._get_collision_force("floor", "foot_FR")[:3],
                              self._get_collision_force("floor", "foot_RL")[:3],
                              self._get_collision_force("floor", "foot_RR")[:3]])

        return grf

    def _set_goal_arrow(self):
        """
        Sets the rotation of the goal arrow based on the current trunk
        rotation angle and the current goal direction.

        """

        desired_angle = self._goal.get_direction()

        rot_mat = euler_to_mat(np.array([np.pi/2, 0, desired_angle]))

        # and set the rotation of the cylinder
        self._data.site("dir_arrow").xmat = (rot_mat).reshape((9,))

        # calc position of the ball corresponding to the arrow
        self._data.site("dir_arrow_ball").xpos = self._data.body("dir_arrow").xpos + \
                                                 [-0.1 * np.sin(desired_angle), 0.1 * np.cos(desired_angle), 0]

    def _init_sim_from_obs(self, obs):
        """
        Initializes the simulation from an observation.

        Args:
            obs (np.array): The observation to set the simulation state to.

        """
        raise TypeError("Initializing from observation is currently not supported in this environment. ")

    def _get_interpolate_map_params(self):
        """
        Returns all parameters needed to do the interpolation mapping for the respective environment.

        """
        keys = self.get_all_observation_keys()
        rot_mat_idx = keys.index("dir_arrow")
        trunk_list_idx = keys.index("q_trunk_list")
        trunk_tilt_idx = keys.index("q_trunk_tilt")
        trunk_rot_idx = keys.index("q_trunk_rotation")

        return dict(rot_mat_idx=rot_mat_idx, trunk_orientation_idx=[trunk_list_idx, trunk_tilt_idx, trunk_rot_idx])

    def _get_interpolate_remap_params(self):
        """
        Returns all parameters needed to do the interpolation remapping for the respective environment.

        """
        keys = self.get_all_observation_keys()
        angle_idx = keys.index("dir_arrow")
        trunk_list_idx = keys.index("q_trunk_list")
        trunk_tilt_idx = keys.index("q_trunk_tilt")
        trunk_rot_idx = keys.index("q_trunk_rotation")
        position_indices = [keys.index(key) for key in keys if key.startswith("q_")]
        velocity_indices = [keys.index(key) for key in keys if key.startswith("dq_")]
        ctrl_dt = self.dt

        return dict(angle_idx=angle_idx, trunk_orientation_idx=[trunk_list_idx, trunk_tilt_idx, trunk_rot_idx],
                    position_indices=position_indices, velocity_indices=velocity_indices, ctrl_dt=ctrl_dt)

    @staticmethod
    def generate(task="simple", dataset_type="real", debug=False, **kwargs):
        """
        Returns a Unitree environment corresponding to the specified task.

        Args:
            task (str): Main task to solve. Either "simple" or "hard". "simple" is a straight walking
                task. "hard" is a walking task in 8 direction. This makes this environment goal conditioned.
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.

        Returns:
            An MDP of the Unitree A1 Robot.

        """
        check_validity_task_mode_dataset(UnitreeA1.__name__, task, None, dataset_type,
                                         *UnitreeA1.valid_task_confs.get_all())

        if "reward_type" in kwargs.keys():
            reward_type = kwargs["reward_type"]
            reward_params = kwargs["reward_params"]
            del kwargs["reward_type"]
            del kwargs["reward_params"]
        else:
            reward_type = "velocity_vector"
            reward_params = dict()

        # Generate the MDP
        # todo: once the trajectory is learned without random init rotation, activate the latter.
        if task == "simple":
            if dataset_type == "real":
                path = "datasets/quadrupeds/real/walk_straight.npz"
            elif dataset_type == "perfect":
                path = "datasets/quadrupeds/perfect/unitreea1_simple/perfect_expert_dataset_det.npz"
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)
            mdp = UnitreeA1(reward_type=reward_type, reward_params=reward_params, **kwargs)
            traj_path = Path(loco_mujoco.__file__).resolve().parent / path
        elif task == "hard":
            if dataset_type == "real":
                path = "datasets/quadrupeds/real/walk_8_dir.npz"
            elif dataset_type == "perfect":
                path = "datasets/quadrupeds/perfect/unitreea1_hard/perfect_expert_dataset_det.npz"
            use_mini_dataset = not os.path.exists(Path(loco_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)
            mdp = UnitreeA1(reward_type=reward_type, reward_params=reward_params, **kwargs)
            traj_path = Path(loco_mujoco.__file__).resolve().parent / path

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
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "action_mode" in kwargs.keys():
                assert kwargs["action_mode"] == "torque"
            if "default_target_velocity" in kwargs.keys():
                assert kwargs["default_target_velocity"] == 0.5
            traj_data_freq = 100  # hz
            traj_files = mdp.load_dataset_and_get_traj_files(path, traj_data_freq)
            traj_params = dict(traj_files=traj_files,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq))

        mdp.load_trajectory(traj_params)

        return mdp

    @property
    def _goal_velocity_idx(self):
        """
        The goal speed is not in the observation helper, but only in the trajectory. This is a workaround
        to access it anyways.
        Note: This is the goal velocity in index *before* the modify_observation_callback is applied!

        """
        return 43

    @staticmethod
    def _modify_observation_callback(obs, rot_mat_idx_arrow, goal_velocity_idx):
        """
        Transforms the rotation matrix from obs to a sin-cos feature.

        Args:
            obs (np.array): Generated observation.
            rot_mat_idx_arrow (int): Index of the beginning rotation matrix in the observation.
            goal_velocity_idx (int): Index of the goal speed in the observation.

        Returns:
            The final environment observation for the agent.

        """

        rot_mat_arrow = obs[rot_mat_idx_arrow].reshape((3, 3))
        # convert mat to angle
        angle = mat2angle_xy(rot_mat_arrow)
        # transform the angle to be in [-pi, pi]
        angle = transform_angle_2pi(angle)
        # rotate by 90 degrees
        angle -= np.pi / 2
        # make sin-cos transformation
        angle = np.array([np.cos(angle), np.sin(angle)])

        # get goal velocity
        goal_velocity = obs[goal_velocity_idx]

        # concatenate everything to new obs
        new_obs = np.concatenate([obs[:rot_mat_idx_arrow[0]], angle, [goal_velocity]])

        return new_obs

    @staticmethod
    def _add_dir_vector_to_xml_handle(xml_handle):
        """
        Adds a direction vector to the Mujoco XML visualizing the goal direction.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find trunk and attach direction arrow
        trunk = xml_handle.find("body", "trunk")
        trunk.add("body", name="dir_arrow", pos="0 0 0.15")
        dir_vec = xml_handle.find("body", "dir_arrow")
        # todo: once Mujoco support cones, make an actual arrow (its a requested feature).
        dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".03", pos="-.1 0 0")
        dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            ("q_trunk_tx", "trunk_tx", ObservationType.JOINT_POS),
            ("q_trunk_ty", "trunk_ty", ObservationType.JOINT_POS),
            ("q_trunk_tz", "trunk_tz", ObservationType.JOINT_POS),
            ("q_trunk_list", "trunk_list", ObservationType.JOINT_POS),
            ("q_trunk_tilt", "trunk_tilt", ObservationType.JOINT_POS),
            ("q_trunk_rotation", "trunk_rotation", ObservationType.JOINT_POS),
            # --- Front ---
            ("q_FR_hip_joint", "FR_hip_joint", ObservationType.JOINT_POS),
            ("q_FR_thigh_joint", "FR_thigh_joint", ObservationType.JOINT_POS),
            ("q_FR_calf_joint", "FR_calf_joint", ObservationType.JOINT_POS),
            ("q_FL_hip_joint", "FL_hip_joint", ObservationType.JOINT_POS),
            ("q_FL_thigh_joint", "FL_thigh_joint", ObservationType.JOINT_POS),
            ("q_FL_calf_joint", "FL_calf_joint", ObservationType.JOINT_POS),
            # --- Rear ---
            ("q_RR_hip_joint", "RR_hip_joint", ObservationType.JOINT_POS),
            ("q_RR_thigh_joint", "RR_thigh_joint", ObservationType.JOINT_POS),
            ("q_RR_calf_joint", "RR_calf_joint", ObservationType.JOINT_POS),
            ("q_RL_hip_joint", "RL_hip_joint", ObservationType.JOINT_POS),
            ("q_RL_thigh_joint", "RL_thigh_joint", ObservationType.JOINT_POS),
            ("q_RL_calf_joint", "RL_calf_joint", ObservationType.JOINT_POS),
            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ("dq_trunk_tx", "trunk_tx", ObservationType.JOINT_VEL),
            ("dq_trunk_ty", "trunk_ty", ObservationType.JOINT_VEL),
            ("dq_trunk_tz", "trunk_tz", ObservationType.JOINT_VEL),
            ("dq_trunk_list", "trunk_list", ObservationType.JOINT_VEL),
            ("dq_trunk_tilt", "trunk_tilt", ObservationType.JOINT_VEL),
            ("dq_trunk_rotation", "trunk_rotation", ObservationType.JOINT_VEL),
            # --- Front ---
            ("dq_FR_hip_joint", "FR_hip_joint", ObservationType.JOINT_VEL),
            ("dq_FR_thigh_joint", "FR_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_FR_calf_joint", "FR_calf_joint", ObservationType.JOINT_VEL),
            ("dq_FL_hip_joint", "FL_hip_joint", ObservationType.JOINT_VEL),
            ("dq_FL_thigh_joint", "FL_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_FL_calf_joint", "FL_calf_joint", ObservationType.JOINT_VEL),
            # --- Rear ---
            ("dq_RR_hip_joint", "RR_hip_joint", ObservationType.JOINT_VEL),
            ("dq_RR_thigh_joint", "RR_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_RR_calf_joint", "RR_calf_joint", ObservationType.JOINT_VEL),
            ("dq_RL_hip_joint", "RL_hip_joint", ObservationType.JOINT_VEL),
            ("dq_RL_thigh_joint", "RL_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_RL_calf_joint", "RL_calf_joint", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"]

        return action_spec

    @staticmethod
    def _interpolate_map(traj, **interpolate_map_params):
        """
        A mapping that is supposed to transform a trajectory into a space where interpolation is
        allowed. E.g., maps a rotation matrix to a set of angles.

        Args:
            traj (list): List of np.arrays containing each observations. Each np.array
                has the shape (n_trajectories, n_samples, (dim_observation)). If dim_observation
                is one the shape of the array is just (n_trajectories, n_samples).
            interpolate_map_params: Set of parameters needed to do the interpolation by the Unitree environment.

        Returns:
            A np.array with shape (n_observations, n_trajectories, n_samples). dim_observation
            has to be one.

        """

        rot_mat_idx = interpolate_map_params["rot_mat_idx"]
        trunk_orientation_idx = interpolate_map_params["trunk_orientation_idx"]
        traj_list = [list() for j in range(len(traj))]
        for i in range(len(traj_list)):
            # if the state is a rotation
            if i in trunk_orientation_idx:  # todo: not sure if this is actually needed.
                # change it to the nearest rotation presentation to the previous state
                # -> no huge jumps between -pi and pi for example
                traj_list[i] = list(np.unwrap(traj[i]))
            else:
                traj_list[i] = list(traj[i])
        # turn matrices into angles todo: this is slow, implement vectorized implementation
        traj_list[rot_mat_idx] = np.array([mat2angle_xy(mat) for mat in traj[rot_mat_idx]])
        return np.array(traj_list)

    @staticmethod
    def _interpolate_remap(traj, **interpolate_remap_params):
        """
        The corresponding backwards transformation to _interpolation_map.

        Args:
            traj (np.array): Trajectory as np.array with shape (n_observations, n_trajectories, n_samples).
            dim_observation is one.
            interpolate_remap_params: Set of parameters needed to do the interpolation by the Unitree environment.

        Returns:
            List of np.arrays containing each observations. Each np.array has the shape
            (n_trajectories, n_samples, (dim_observation)). If dim_observation
            is one the shape of the array is just (n_trajectories, n_samples).

        """

        angle_idx = interpolate_remap_params["angle_idx"]
        trunk_orientation_idx = interpolate_remap_params["trunk_orientation_idx"]
        position_indices = interpolate_remap_params["position_indices"]
        velocity_indices = interpolate_remap_params["velocity_indices"]
        ctrl_dt = interpolate_remap_params["ctrl_dt"]
        traj_list = [list() for j in range(len(traj))]
        for i in range(len(traj_list)):
            # if the state is a rotation
            if i in trunk_orientation_idx:
                # make sure it is in range -pi,pi
                traj_list[i] = [transform_angle_2pi(angle) for angle in traj[i]]
            elif i in velocity_indices:
                # the interpolation is problematic in the joint velocities for the Unitree. Recalculate them here based
                # on the positions
                joint_position_idx = position_indices[velocity_indices.index(i)]
                joint_position = traj[joint_position_idx]
                traj_list[i] = [0.0] + list((joint_position[1:] - joint_position[:-1]) / ctrl_dt)
            else:
                traj_list[i] = list(traj[i])

        # transforms angles into rotation matrices todo: this is slow, implement vectorized implementation
        traj_list[angle_idx] = np.array([angle2mat_xy(angle).reshape(9,) for angle in traj[angle_idx]])
        return traj_list
