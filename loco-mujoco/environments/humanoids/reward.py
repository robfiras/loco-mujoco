from pathlib import Path
import pickle
from itertools import cycle

import numpy as np

from mushroom_rl.environments.mujoco_envs.humanoids.utils import convert_traj_quat_to_euler

from mushroom_rl.utils.running_stats import *

# todo: remove this class once refactoring is done
class BaseHumanoidTrajectory:
    def __init__(self):
        pass

class GoalRewardInterface:
    """
    Interface to specify a reward function for the ``HumanoidGait`` environment.

    """
    def __call__(self, state, action, next_state):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returs:
            The reward for the current transition.

        """
        raise NotImplementedError

    def get_observation_space(self):
        """
        Getter.

        Returns:
             The low and high arrays of the observation space

        """
        obs = self.get_observation()
        return -np.inf * np.ones(len(obs)), np.inf * np.ones(len(obs))

    def get_observation(self):
        """
        Getter.

        Returns:
             The current observation.

        """
        return np.array([])

    def is_absorbing(self, state):
        """
        Getter.

        Returns:
            Whether the current state is absorbing or not.

        """
        return False

    def update_state(self):
        """
        Update the state of the object after each transition.

        """
        pass

    def reset_state(self):
        """
        Reset the state of the object.

        """
        pass


class NoGoalReward(GoalRewardInterface):
    """
    Implement a reward function that is always 0.

    """

    def __call__(self, state, action, next_state):
        return 0


class NoGoalRewardRandInit(GoalRewardInterface, BaseHumanoidTrajectory):
    """
    Implement a reward function that is always 0, and starts from random positions.

    """

    def __call__(self, state, action, next_state):
        return 0

    def reset_state(self):
        self.reset_trajectory()
        self.terminate_trajectory_flag = False


class CustomReward(GoalRewardInterface):

    def __init__(self, reward_callback=None, random_start=True, random_start_point=None,  **kwargs):
        self._reward_callback = reward_callback

    def __call__(self, state, action, next_state):
        if self._reward_callback is not None:
            return self._reward_callback(state, action, next_state)
        else:
            0


class TargetVelocityReward(GoalRewardInterface):

    def __init__(self, target_velocity, x_vel_idx):
        self._target_vel = target_velocity
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state):
        x_vel = state[self._x_vel_idx]
        return np.exp(- np.square(x_vel - self._target_vel))


class MultTargetVelocityReward(GoalRewardInterface):

    def __init__(self, target_velocity, x_vel_idx, env_id_len, scalings):
        self._target_vel = target_velocity
        self._env_id_len = env_id_len
        self._scalings = scalings
        self._x_vel_idx = x_vel_idx

    def __call__(self, state, action, next_state):
        x_vel = state[self._x_vel_idx]
        env_id = state[-self._env_id_len:]

        # convert binary array to index
        ind = np.packbits(env_id.astype(int), bitorder='big') >> (8 - env_id.shape[0])
        ind = ind[0]
        scaling = self._scalings[ind]

        # calculate target vel
        target_vel = self._target_vel * scaling

        return np.exp(- np.square(x_vel - target_vel))


class PosReward(GoalRewardInterface):

    def __init__(self, pos_idx):
        self._pos_idx = pos_idx

    def __call__(self, state, action, next_state):
        pos = state[self._pos_idx]
        return pos


class ChangingVelocityTargetReward(BaseHumanoidTrajectory, GoalRewardInterface):

    def __init__(self, sim, traj_path, goal_data_path, iterate_through_plateaus=False, silent=True, traj_dt=0.005,
                 control_dt=0.005, traj_speed_mult=1.0, activate_keyboard_controller=False,
                 velocity_smooth_window=1001, random_start=True, n_skip_targets=1):

        super().__init__(sim, traj_path, traj_dt, control_dt, traj_speed_mult,
                         velocity_smooth_window)

        # get the goal data
        goal_file = open(goal_data_path, 'rb')
        goal_data = pickle.load(goal_file)
        assert goal_data["type"] == "ChangingVelocityTargetReward", "Need ChangingVelocityTargetReward" \
                                                                    "but found %s." % goal_data["type"]
        self._goal_velocities = goal_data["vel"]
        self._goal_ind_chunks = goal_data["vel_ind_chunks"]
        self._goal_plateaus = goal_data["plateau_vel"]
        self._curr_goal_vel = 0.0
        self._random_start = random_start
        self._iterate_through_plateaus = iterate_through_plateaus
        self._n_skip_targets = n_skip_targets
        self._iter_plateaus = cycle(np.arange(len(self._goal_plateaus))) if iterate_through_plateaus else None
        self._silent = silent
        self.mean_vel = RunningExpWeightedAverage(shape=(1,), alpha=0.005)

        # activate keyboard controller
        if activate_keyboard_controller:
            pass
            #self._curr_goal_vel_queue = Queue()
            #self._keyboard_reader = Process()



    def __call__(self, state, action, next_state):
        curr_v = state[16]  # velocity in y
        v_diff_rew = np.minimum(0.01 / (np.abs(curr_v - self._curr_goal_vel) + 1e-6), 1.0)
        self.mean_vel.update_stats(curr_v)
        if not self._silent:
            print("Current_Velocity: ", self.mean_vel.mean, " Target: ", self._curr_goal_vel, " Abs. Difference: ",
                  np.abs(self.mean_vel.mean - self._curr_goal_vel))
        return v_diff_rew

    def get_observation(self):
        return [self._curr_goal_vel]

    def reset_state(self):
        if not self._iterate_through_plateaus:
            # sample random plateau vel
            ind_plateau = np.random.randint(len(self._goal_plateaus))
        else:
            for i in range(self._n_skip_targets):
                ind_plateau = next(self._iter_plateaus)
        if self._random_start:
            # sample random start fos this plateau
            ind_state_in_chunk = np.random.randint(len(self._goal_ind_chunks[ind_plateau]))
            ind_state = self._goal_ind_chunks[ind_plateau][ind_state_in_chunk]
            self._curr_goal_vel = self._goal_plateaus[ind_plateau]
            self.reset_trajectory(ind_state)
            self.mean_vel.reset(self._curr_goal_vel)
        else:
            self._curr_goal_vel = self._goal_plateaus[ind_plateau]
        if not self._silent:
            print("Current Target Velocity: ", self._curr_goal_vel)


class MaxVelocityReward(GoalRewardInterface):
    """
    Implement a goal reward for achieving the maximum possible velocity.

    """
    def __init__(self, sim, traj_start=False, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            traj_start (bool, False): If model initial position should be set
                from a valid trajectory state. If False starts from the
                model.xml base position;
            **kwargs: additional parameters which can be passed to
                trajectory when using ``traj_start``. ``traj_path`` should be
                given to select a different trajectory. Rest of the arguments
                are not important.

        """
        self.traj_start = traj_start

        if traj_start:
            if "traj_path" not in kwargs:
                traj_path = Path(__file__).resolve().parent.parent.parent /\
                            "data" / "humanoid_gait" / "gait_trajectory.npz"
                kwargs["traj_path"] = traj_path.as_posix()
            self.trajectory = BaseHumanoidTrajectory(sim, **kwargs)
            self.reset_state()

    def __call__(self, state, action, next_state):
        return next_state[13]

    def reset_state(self):
        if self.traj_start:
            self.trajectory.reset_trajectory()


class VelocityProfileReward(GoalRewardInterface):
    """
    Implement a goal reward for following a velocity profile.

    """
    def __init__(self, sim, profile_instance, traj_start=False, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            profile_instance (VelocityProfile): Velocity profile to
                follow. See RewardGoals.velocity_profile.py;
            traj_start (bool, False): If model initial position should be set
                from a valid trajectory state. If False starts from the
                model.xml base position;
            **kwargs: additional parameters which can be passed to
                trajectory when using ``traj_start``. ``traj_path`` should be
                given to select a diferent trajectory. Rest of the arguments
                are not important.

        """
        self.profile = profile_instance
        self.velocity_profile = deque(self.profile.values)

        self.traj_start = traj_start
        if traj_start:
            if "traj_path" not in kwargs:
                traj_path = Path(__file__).resolve().parent.parent.parent /\
                            "data" / "humanoid_gait" / "gait_trajectory.npz"
                kwargs["traj_path"] = traj_path.as_posix()
            self.trajectory = BaseHumanoidTrajectory(sim, **kwargs)
            self.reset_state()

    def __call__(self, state, action, next_state):
        return np.exp(-np.linalg.norm(next_state[13:16] - self.velocity_profile[0]))

    def update_state(self):
        self.velocity_profile.rotate(1)

    def get_observation(self):
        return self.velocity_profile[0]

    def reset_state(self):
        self.velocity_profile = deque(self.profile.reset())

        if self.traj_start:
            substep_no = np.argmin(
                    np.linalg.norm(
                            self.trajectory.velocity_profile
                            - np.expand_dims(self.velocity_profile[0], 1),
                            axis=0))
            self.trajectory.reset_trajectory(substep_no=substep_no)

    def plot_velocity_profile_example(self, horizon=1000, n_trajectories=10):
        values = np.zeros((horizon * n_trajectories, 3))
        i = 0
        for t in range(n_trajectories):
            self.reset_state()
            for s in range(horizon):
                self.update_state()
                values[i, :] = self.get_observation()
                i += 1

        self.reset_state()
        import matplotlib.pyplot as plt
        plt.plot(values)
        for line_pos in [i*horizon for i in range(n_trajectories)]:
            plt.axvline(line_pos, c="red", alpha=0.3)
        plt.legend(["axis x", "axis y", "axis z"])
        plt.show()


class CompleteTrajectoryReward(GoalRewardInterface, BaseHumanoidTrajectory):
    """
    Implements a goal reward for matching a kinematic trajectory.

    """
    def __init__(self, sim, control_dt=0.005, traj_path=None,
                 traj_dt=0.0025, traj_speed_mult=1.0,
                 use_error_terminate=False, joint_pos_rew_weight=1.0,
                 joint_vel_rew_weight=1.0, com_pos_rew_weight=0.1, com_vel_rew_weight=0.1, com_pos_only_z=True,
                 feet_pos_rew_weight=1.0, rew_scale=1.0, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            control_dt (float, 0.005): frequency of the controller;
            traj_path (string, None): path with the trajectory for the
                model to follow. If None is passed, use default
                trajectory;
            traj_dt (float, 0.0025): time step of the trajectory file;
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity;
            use_error_terminate (bool, False): If episode should be terminated
                when the model deviates significantly from the reference
                 trajectory.

        """
        if traj_path is None:
            traj_path = Path(__file__).resolve().parent.parent.parent / "data" / "humanoid_gait" / "gait_trajectory.npz"
            traj_path = traj_path.as_posix()

        super(CompleteTrajectoryReward, self).__init__(sim, traj_path, traj_dt,
                                                       control_dt,
                                                       traj_speed_mult)
        self.error_terminate = use_error_terminate

        self._joint_pos_rew_weight = joint_pos_rew_weight
        self._joint_vel_rew_weight = joint_vel_rew_weight
        self._com_pos_rew_weight = com_pos_rew_weight
        self._com_vel_rew_weight = com_vel_rew_weight
        self._com_pos_only_z = com_pos_only_z
        self._feet_pos_rew_weight = feet_pos_rew_weight
        self._rew_scale = rew_scale

        self.error_threshold = 0.20
        self.terminate_trajectory_flag = False

        self.euler_traj = convert_traj_quat_to_euler(self.subtraj)

        self.traj_data_range = np.clip(2 * np.std(self.euler_traj, axis=1), 0.15, np.inf)

        self.joint_importance = np.where(self.traj_data_range < 0.15, 2 * self.traj_data_range, 1.0)
        self.joint_importance[2:14] *= 1.0
        self.joint_importance[17:28] *= 0.1
        self.joint_importance[28:34] *= 5.0
        self.joint_importance = np.r_[self.joint_importance[2:14],
                                      self.joint_importance[17:28],
                                      self.joint_importance[28:34]]

        self.traj_data_range = np.concatenate([self.traj_data_range[2:14],
                                               self.traj_data_range[17:34]])

    def __call__(self, state, action, next_state):
        rew_com_pos, rew_joint_pos, rew_com_vel,\
        rew_joint_vel, rew_foot_pos = self._calculate_each_comp_reward(state, action, next_state)

        reward = self._com_pos_rew_weight * rew_com_pos\
                 + self._joint_pos_rew_weight * rew_joint_pos\
                 + self._com_vel_rew_weight * rew_com_vel\
                 + self._joint_vel_rew_weight * rew_joint_vel \
                 + self._feet_pos_rew_weight * rew_foot_pos

        return reward * self._rew_scale

    def _calculate_each_comp_reward(self, state, action, next_state):

        # --- get current data ---
        euler_state = convert_traj_quat_to_euler(next_state, offset=0)
        if self._com_pos_only_z:
            current_com_pos = euler_state[2:6]
        else:
            current_com_pos = euler_state[0:6]
        current_q_pos = euler_state[6:16]
        current_com_vel = euler_state[16:22]
        current_q_vel = euler_state[22:32]

        rel_foot_vec = np.append(
            (self.sim.data.get_body_xpos("torso") - self.sim.data.get_body_xpos("right_foot")),
            (self.sim.data.get_body_xpos("torso") - self.sim.data.get_body_xpos("left_foot"))
        )

        # --- get target data ---
        if self._com_pos_only_z:
            tar_com_pos = self.euler_traj[2:6, self.subtraj_step_no]
        else:
            tar_com_pos = self.euler_traj[0:6, self.subtraj_step_no]
        tar_q_pos = self.euler_traj[6:16, self.subtraj_step_no]
        tar_com_vel = self.euler_traj[16:22, self.subtraj_step_no]
        tar_q_vel = self.euler_traj[22:32, self.subtraj_step_no]

        # get the start and end of the foot positions in an expert state
        foot_vec_start = np.squeeze(np.where(np.array(self.keys) == "rel_feet_xpos_r")).item() - 1
        foot_vec_end = np.squeeze(np.where(np.array(self.keys) == "rel_feet_zpos_l")).item()
        tar_foot_vec = self.euler_traj[foot_vec_start:foot_vec_end, self.subtraj_step_no]

        # --- calculate all rewards ---
        rew_com_pos = np.exp(-5.0 * np.mean(np.square(current_com_pos - tar_com_pos)))
        rew_joint_pos = np.exp(-2.0 * np.mean(np.square(current_q_pos - tar_q_pos)))
        rew_com_vel = np.exp(-5.0 * np.mean(np.square(current_com_vel - tar_com_vel)))
        rew_joint_vel = np.exp(-0.1 * np.mean(np.square(current_q_vel - tar_q_vel)))
        rew_foot_pos = np.exp(-40.0 * np.mean(np.square(rel_foot_vec - tar_foot_vec)))

        return rew_com_pos, rew_joint_pos, rew_com_vel, rew_joint_vel, rew_foot_pos

    def update_state(self):
        self.subtraj_step_no += 1
        if self.subtraj_step_no >= self.traj_length:
            self.get_next_sub_trajectory()

    def get_observation(self):
        return self.velocity_profile[:, self.subtraj_step_no]

    def is_absorbing(self, state):
        return self.terminate_trajectory_flag

    def reset_state(self):
        self.reset_trajectory()
        self.terminate_trajectory_flag = False
