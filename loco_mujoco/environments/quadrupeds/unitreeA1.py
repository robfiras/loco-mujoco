import os
import time
import sys
from abc import abstractmethod
import mujoco
from dm_control import mjcf
from copy import deepcopy

from pathlib import Path
from scipy import interpolate

import numpy as np
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.environments.mujoco_envs.humanoids.trajectory import Trajectory

import matplotlib.pyplot as plt
import random

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward

from loco_mujoco import BaseEnv

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class UnitreeA1(BaseEnv):
    """
    Mujoco simulation of unitree A1 model
    """

    def __init__(self, gamma=0.99, horizon=1000, n_substeps=10, timestep=0.001, random_start=False,
                 init_step_no=None, init_traj_no=None, traj_params=None, goal_reward=None, goal_reward_params=None,
                 use_torque_ctrl=True, use_2d_ctrl=False, tmp_dir_name=None, setup_random_rot=False):
        """
        Constructor.
        Args:
            gamma (float): The discounting factor of the environment;
            horizon (int): The maximum horizon for the environment;
            n_substeps (int, 1): The number of substeps to use by the MuJoCo
                simulator. An action given by the agent will be applied for
                n_substeps before the agent receives the next observation and
                can act accordingly;
            timestep (float): The timestep used by the MuJoCo
                simulator. If None, the default timestep specified in the XML will be used;
            random_start (bool): if the robot should start in a random state from self.trajectory
            init_step_no (int): if not random start in which step no the robot should start
            init_traj_no (int): no of trajectory if multiple trajectories are in self.trajectory
            traj_params (list): list of parameters for the trajectory class
            goal_reward (str): how the reward should be calculated; options: "custom", "target_velocity", None
            goal_reward_params (list): parameters for a custom goal reward
            use_torque_ctrl (bool): if the unitree should use torque or position control
            use_2d_ctrl (bool): if the goal is to walk in two dimensions: will add a direction arrow to distinguish
                between different directions
            tmp_dir_name (str): path to a temporary directory to store the temporary new xml file with the
                direction arrow
            setup_random_rot (bool): if the robot should be set up with a random rotation
        """
        # different xml files for torque and position control
        if use_torque_ctrl:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_torque.xml").as_posix()
            xml_path = "/home/moore/PycharmProjects/loco-mujoco/loco_mujoco/environments/data/quadrupeds"
            print("Using torque-control for unitreeA1")
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_position.xml").as_posix()
            print("Using position-control for unitreeA1")

        # motors
        action_spec = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"]
        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            ("q_trunk_tx", "trunk_tx", ObservationType.JOINT_POS),
            ("q_trunk_ty", "trunk_ty", ObservationType.JOINT_POS),
            ("q_trunk_tz", "trunk_tz", ObservationType.JOINT_POS),
            ("q_trunk_rotation", "trunk_rotation", ObservationType.JOINT_POS),
            ("q_trunk_list", "trunk_list", ObservationType.JOINT_POS),
            ("q_trunk_tilt", "trunk_tilt", ObservationType.JOINT_POS),
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
            ("dq_trunk_rotation", "trunk_rotation", ObservationType.JOINT_VEL),
            ("dq_trunk_list", "trunk_list", ObservationType.JOINT_VEL),
            ("dq_trunk_tilt", "trunk_tilt", ObservationType.JOINT_VEL),
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

        # important contact forces
        collision_groups = [("floor", ["floor"]),
                            ("foot_FR", ["FR_foot"]),
                            ("foot_FL", ["FL_foot"]),
                            ("foot_RR", ["RR_foot"]),
                            ("foot_RL", ["RL_foot"])]

        if use_2d_ctrl:
            # append observation_spec with the direction arrow & add it to a temporary xml file
            observation_spec.append(("dir_arrow", "dir_arrow", ObservationType.SITE_ROT))
            assert tmp_dir_name is not None, "If you want to use 2d_ctrl, you have to specify a" \
                                             "directory name for the xml-files to be saved."
            xml_handle = self.add_dir_vector_to_xml_handle(mjcf.from_path(xml_path))
            xml_path = self.save_xml_handle(xml_handle, tmp_dir_name)
        self.use_2d_ctrl = use_2d_ctrl
        self.setup_random_rot = setup_random_rot

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                         timestep=timestep, collision_groups=collision_groups, traj_params=traj_params,
                         init_step_no=init_step_no, init_traj_no=init_traj_no, goal_reward=goal_reward,
                         goal_reward_params=goal_reward_params, random_start=random_start)

    def _modify_observation(self, obs, dir_arrow_from_robot_pov=False):
        """
        transform the rotation from the simulation to the observation we need for training:
        transform rotation matrix of the direction arrow into sind and cos of the corresponding angle

        Args:
            obs (np.ndarray): the generated observation
            dir_arrow_from_robot_pov (Bool): if the rotation angle is from the robot pov or in absolute coordinates

        Returns:
            The environment observation.

        """
        if self.use_2d_ctrl:
            new_obs = obs[:34]
            # transform rotation matrix into rotation angle
            temp = np.dot(obs[34:43].reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
            # depending on the point of view substract the trunk rotation
            if dir_arrow_from_robot_pov:
                angle = (np.arctan2(temp[3], temp[0]) + np.pi) % (2 * np.pi) - np.pi
            else:
                angle = (np.arctan2(temp[3], temp[0]) - obs[1] + np.pi) % (2 * np.pi) - np.pi
            # and turn angle to sin, cos (for a closed angle range)
            new_obs = np.append(new_obs, [np.cos(angle), np.sin(angle)])
            new_obs = np.append(new_obs, obs[43:])
            return new_obs
        return obs

    def add_dir_vector_to_xml_handle(self, xml_handle):
        """
        add the xml elements for the direction arrow
        """
        # find trunk and attach direction arrow
        trunk = xml_handle.find("body", "trunk")
        trunk.add("body", name="dir_arrow", pos="0 0 0.15")
        dir_vec = xml_handle.find("body", "dir_arrow")
        dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".03", pos="-.1 0 0")
        dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="-.1 0 0 .1 0 0")

        return xml_handle

    def save_xml_handle(self, xml_handle, tmp_dir_name):
        """
        same new xml file with the direction arrow in tmp_dir_name
        """
        # save new model and return new xml path
        new_model_dir_name = 'new_unitree_a1_with_dir_vec_model/' + tmp_dir_name + "/"
        cwd = Path.cwd()
        new_model_dir_path = Path.joinpath(cwd, new_model_dir_name)
        xml_file_name = "modified_unitree_a1.xml"
        mjcf.export_with_assets(xml_handle, new_model_dir_path, xml_file_name)
        new_xml_path = Path.joinpath(new_model_dir_path, xml_file_name)

        return new_xml_path.as_posix()

    def setup(self, substep_no=None):
        """
        sets up the initial state of the robot
        """

        self.goal_reward.reset_state()
        if self.trajectory is not None:
            if self._random_start:
                # choose random state from trajectory
                sample = self.trajectory.reset_trajectory()
            else:
                # choose specified state
                sample = self.trajectory.reset_trajectory(self._init_step_no, self._init_traj_no)
            # if we want to set up the robot with a random rotation
            if self.setup_random_rot:
                angle = np.random.uniform(0, 2 * np.pi)
                sample = rotate_obs(sample, angle, False)

            # if we use the direction arrow
            if self.use_2d_ctrl:
                # turn matrix into angle
                mat = np.dot(sample[36].reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                angle = np.arctan2(mat[3], mat[0])
                # _goals contains two lists: first the list of goal states already represented in the observation_spec \
                #   (direction angle), and the list of goal states that are not in obs_spec (velo) # todo double check on the latter
                self._goals = np.array([[angle], [sample[37]]], dtype=float)

                # add the trunk rotation to the direction to make sure the direction arrow matrix is interpreted from \
                # the robot's point of view
                trunk_rotation = sample[3]
                # calc rotation matrix with rotation of trunk
                R = np.array(
                    [[np.cos(trunk_rotation), -np.sin(trunk_rotation), 0],
                     [np.sin(trunk_rotation), np.cos(trunk_rotation), 0],
                     [0, 0, 1]])
                sample[36] = np.dot(R, sample[36].reshape((3, 3))).reshape((9,))
            # set state
            self.set_qpos_qvel(sample)
        else:  # TODO: because init position from xml is bad
            # defines default position
            sample = [0, 0, -0.16, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if self.setup_random_rot:
                angle = np.random.uniform(0, 2 * np.pi)
                sample = rotate_obs(sample, angle, False)
            len_qpos, len_qvel = self.len_qpos_qvel()
            self._data.qpos = sample[0:len_qpos]
            self._data.qvel = sample[len_qpos:len_qpos + len_qvel]
            if self.use_2d_ctrl:
                # _goals contains two lists: first the list of goal states already represented in the observation_spec \
                #   (direction angle), and the list of goal states that are not in obs_spec (velo)
                self._goals = np.array([[0], [0]], dtype=float)

    @staticmethod
    def has_fallen(state):
        """
        return if the robot has fallen in state
        """
        trunk_euler = state[1:4]
        trunk_condition = ((trunk_euler[1] < -0.2793) or (trunk_euler[1] > 0.2793)
                           # max x-rotation 11 degree -> accepts 16 degree
                           or (trunk_euler[2] < -0.192) or (trunk_euler[2] > 0.192)
                           # max y-rotation 7.6 deg -> accepts 11 degree
                           or state[0] < -.24
                           # min height -0.197 -> accepts 0.24
                           )
        return trunk_condition

    def _simulation_post_step(self):
        """
        what to do before the simulation step:
            update ground forces
            and set the position of the direction arrow corresponding to the angle in goals[0,0]
        """
        grf = np.concatenate([self._get_collision_force("floor", "foot_FL")[:3],
                              self._get_collision_force("floor", "foot_FR")[:3],
                              self._get_collision_force("floor", "foot_RL")[:3],
                              self._get_collision_force("floor", "foot_RR")[:3]])

        self.mean_grf.update_stats(grf)

        # adapt orientation of the goal arrow
        if self.use_2d_ctrl:
            # read rotation of trunk
            trunk_rotation = self._data.joint("trunk_rotation").qpos[0]
            desired_rot = self._goals[0, 0] + trunk_rotation
            # calc rotation matrix with rotation of trunk
            rot_matrix = np.array(
                [[np.cos(desired_rot), -np.sin(desired_rot), 0], [np.sin(desired_rot), np.cos(desired_rot), 0],
                 [0, 0, 1]])
            # rotate with the default arrow rotation (else the arrow is vertical)
            arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
            dir_arrow = np.dot(rot_matrix, arrow).reshape((9,))

            # and set the rotation of the cylinder
            self._data.site("dir_arrow").xmat = dir_arrow
            # calc position of the ball corresponding to the arrow
            self._data.site("dir_arrow_ball").xpos = self._data.body("dir_arrow").xpos + \
                                                     [-0.1 * np.cos(desired_rot), -0.1 * np.sin(desired_rot), 0]

    def create_dataset(self, actions_path=None, ignore_keys=[], normalizer=None, only_state=True, use_next_states=True):
        """
        creates dataset for learning with the states stored in self.trajectory
        Args:
            actions_path: path to actions, if not None only_states must be false
            ignore_keys: which keys to ignore
            normalizer: to normalize the states
            only_state: if the training is with only states
            use_next_states: if the dataset should consider next states
        Returns:
            dictionary with the desired datasets
        """

        trajectories = deepcopy(self.trajectory.trajectory)

        # check for has_fallen violations

        for i in range(len(trajectories[0])):
            try:
                transposed = np.transpose(trajectories[2:, i])
                has_fallen_violation = next(x for x in transposed if self.has_fallen(x))
                np.set_printoptions(threshold=sys.maxsize)
                raise RuntimeError("has_fallen violation occured: ", has_fallen_violation)
            except StopIteration:
                pass
        print("No has_fallen violation found")
        # could be helpfull for a new dataset if the ranges in has fallen are still good
        # print("   Traj minimal height:", min([min(trajectories[2][i]) for i in range(len(trajectories[0]))]))
        # print("   Traj max x-rotation:",
        #      max([max(trajectories[4][i], key=abs) for i in range(len(trajectories[0]))], key=abs))
        # print("   Traj max y-rotation:",
        #      max([max(trajectories[5][i], key=abs) for i in range(len(trajectories[0]))], key=abs))

        # remove ignore_keys unimportant for training
        ignore_index = [self.trajectory.keys.index(ikey) for ikey in ignore_keys]
        for idx in sorted(ignore_index, reverse=True):
            trajectories = np.delete(trajectories, idx, 0)

        # apply modify obs to every sample to make sure to store the same obs as in training
        obs_dim = len(self._modify_observation(np.hstack(trajectories[:, 0, 0]), dir_arrow_from_robot_pov=True))
        new_trajectories = np.empty((obs_dim, trajectories.shape[1], trajectories.shape[2]))
        for i in range(len(trajectories[0])):
            transposed_traj = np.transpose(trajectories[:, i])
            for j in range(len(transposed_traj)):
                flat_sample = np.hstack(transposed_traj[j])
                new_trajectories[:, i, j] = self._modify_observation(flat_sample, dir_arrow_from_robot_pov=True)
        trajectories = new_trajectories

        new_states = []
        new_next_states = []
        # for each trajectory in trajectories append to the result vars
        for i in range(len(trajectories[0])):
            trajectory = trajectories[:, i]
            states = np.transpose(trajectory)

            # normalize if needed
            if normalizer:
                normalizer.set_state(dict(mean=np.mean(states, axis=0),
                                          var=1 * (np.std(states, axis=0) ** 2),
                                          count=1))
                states = np.array([normalizer(st) for st in states])

            # to obtain next states: shift the dataset by one
            new_states += list(states[:-1])
            new_next_states += list(states[1:])

        # if actions are also needed
        if not only_state:
            # read values
            trajectory_files_actions = np.load(actions_path, allow_pickle=True)
            trajectory_files_actions = {k: d for k, d in trajectory_files_actions.items()}
            # 3 dim matrix: (no of trajectories, no of samples per trajectory, length of action space)
            interpolate_factor = self.trajectory.traj_dt / self.trajectory.control_dt
            # *1/interpolate_factor because if we need to interpolate, split_points is already interpolated

            trajectories_actions = np.array([list(trajectory_files_actions["action"]
                                                  [int(self.trajectory.split_points[i] * 1 / interpolate_factor + i):
                                                   int(self.trajectory.split_points[
                                                           i + 1] * 1 / interpolate_factor + i + 1)])
                                             for i in range(len(self.trajectory.split_points) - 1)], dtype=object)
            # interpolate if neccessary
            if self.trajectory.traj_dt != self.trajectory.control_dt:
                interpolated_trajectories = [list() for x in range(len(trajectories_actions))]
                for i in range(len(trajectories_actions)):
                    trajectory = trajectories_actions[i]
                    length = len(trajectory)
                    x = np.arange(length)
                    x_new = np.linspace(0, length - 1, round(length * interpolate_factor), endpoint=True)
                    interpolated_trajectories[i] = interpolate.interp1d(x, trajectory, kind="cubic", axis=0)(x_new)
                trajectories_actions = np.array(interpolated_trajectories)
            # leave out last action of every trajectory to be even with the states
            new_actions = []
            for i in range(len(trajectories_actions)):
                new_actions += list(trajectories_actions[i, :-1])

        # should be always false for expert data/checked for has_fallen
        absorbing = np.zeros(len(new_states))

        # depending on the dataset we need, return dictionary
        if only_state and use_next_states:
            print("Using only states and next_states for training")
            return dict(states=np.array(new_states), next_states=np.array(new_next_states), absorbing=absorbing)
        elif not only_state and not use_next_states:
            print("Using states and actions for training")
            return dict(states=np.array(new_states), actions=np.array(new_actions), absorbing=absorbing)
        elif not only_state and use_next_states:
            print("Using states, next_states and actions for training")
            return dict(states=np.array(new_states), next_states=np.array(new_next_states),
                        actions=np.array(new_actions), absorbing=absorbing)
        else:
            raise NotImplementedError("Wrong input or method doesn't support this type now")

    # in commit preprocess expert data: to simulate the actions and get states in mujoco to corresponding states;
    # cut off initial few samples -> but is not up to date/needs adaptions

    def play_action_demo(self, actions_path,
                         use_rendering=True, use_pd_controller=False, interpolate_map=None, interpolate_remap=None):
        """

        Plays a demo of the loaded actions by using the actions in actions_path.
        Args:
            actions_path: path to the .npz file. Should be in format (number of samples/steps, action dimension)
            use_rendering: if the mujoco simulation should be rendered
            use_pd_controller: if the actions should be calculated by a pd controller depending on the positions
            interpolate_map: used for interpolation of the states
            interpolate_remap: used for interpolation of the states
        Returns:
            observed states and performed actions

        """
        # set init step and traj no if set; else choose it random
        traj_no = self._init_traj_no if self._init_traj_no is not None else \
            int(np.random.rand() * len(self.trajectory.trajectory[0]))
        step_no = self._init_step_no if self._init_step_no is not None else \
            int(np.random.rand() * (self.trajectory.traj_length[traj_no] * 0.45))

        # used for initial states
        trajectory = deepcopy(self.trajectory.trajectory[:, traj_no])

        demo_dt = self.trajectory.traj_dt
        control_dt = self.trajectory.control_dt
        interpolate_factor = demo_dt / control_dt

        # load actions
        action_files = np.load(actions_path, allow_pickle=True)
        action_files = {k: d for k, d in action_files.items()}

        # *1/interpolate_factor because if interpolation is needed, split_points is already interpolated
        actions = np.array([list(action_files[key])[int(self.trajectory.split_points[traj_no] * 1 / interpolate_factor):
                                                    int(self.trajectory.split_points[
                                                            traj_no + 1] * 1 / interpolate_factor)]
                            for key in action_files.keys()], dtype=object)[0]

        # interpolate actions
        if demo_dt != control_dt:
            x = np.arange(actions.shape[0])
            x_new = np.linspace(0, actions.shape[0] - 1, round(actions.shape[0] * interpolate_factor),
                                endpoint=True)
            actions = interpolate.interp1d(x, actions, kind="cubic", axis=0)(x_new)

        # set x and y to 0: be carefull need to be at index 0,1
        trajectory[0, :] -= trajectory[0, step_no]
        trajectory[1, :] -= trajectory[1, step_no]

        # set initial position
        self.set_qpos_qvel(trajectory[:, step_no])

        # to return the states & actions
        actions_dataset = []
        states_dataset = [list() for j in range(len(self.obs_helper.observation_spec))]
        assert len(states_dataset) == len(self.obs_helper.observation_spec)

        # for kp controller if needed
        e_old = 0
        # perform actions
        for i in np.arange(actions.shape[0] - 1):
            # choose actions of dataset or pd-controller
            if not use_pd_controller:
                action = actions[i]
            else:
                e = trajectory[6:18, i + 1] - self._data.qpos[6:]
                de = e - e_old
                """
                kp = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
                kd = np.array([1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2])
                """
                kp = 10
                # kd defined the damping of the joints in the xml file
                hip = 1
                rest = 2
                kd = np.array([hip, rest, rest, hip, rest, rest, hip, rest, rest, hip, rest, rest])
                action = kp * e + (kd / control_dt) * de
                e_old = e

            # store actions and states for datasets
            actions_dataset.append(list(action))
            q_pos_vel = list(self._data.qpos[:]) + list(self._data.qvel[:])
            if self.use_2d_ctrl:
                q_pos_vel.append(list(trajectory[36, i]))  #
            for i in range(len(states_dataset)):
                states_dataset[i].append(q_pos_vel[i])

            # preform action
            nstate, _, absorbing, _ = self.step(action)
            if use_rendering:
                self.render()

        return np.array(states_dataset, dtype=object), np.array(actions_dataset)


def rotate_obs(state, angle, modified=True):
    """
    rotates a set of states with angle
    """
    rotated_state = np.array(state).copy()
    # different indizes for a modified obs or a normal obs
    if modified:
        idx_rot = 1
        idx_xvel = 16
        idx_yvel = 17
    else:
        idx_rot = 3
        idx_xvel = 18
        idx_yvel = 19
    # add rotation to trunk rotation and transform to range -np.pi,np.pi
    rotated_state[idx_rot] = (np.array(np.array(state)[idx_rot]) + angle + np.pi) % (2 * np.pi) - np.pi
    # rotate velo x,y
    rotated_state[idx_xvel] = np.cos(angle) * np.array(np.array(state)[idx_xvel]) - np.sin(angle) * np.array(
        np.array(state)[idx_yvel])
    rotated_state[idx_yvel] = np.sin(angle) * np.array(np.array(state)[idx_xvel]) + np.cos(angle) * np.array(
        np.array(state)[idx_yvel])
    return rotated_state


def test_rotate_data(traj_path, rotation_angle, store_path='./new_unitree_a1_with_dir_vec_model'):
    """
    tests the rotation of a dataset:
        adapts the first dataset in traj_path so it looks like the data from training (modified_observation),
        applies rotation and transforms the rotated data back to the normal obs space
        so we can simulate it with play_trajectory_demo
    Returns:
        path to the dataset rotated with the angle
    """
    # load data
    trajectory_files = np.load(traj_path, allow_pickle=True)
    trajectory_files = {k: d for k, d in trajectory_files.items()}
    keys = list(trajectory_files.keys())
    if "split_points" in trajectory_files.keys():
        split_points = trajectory_files["split_points"]
        keys.remove("split_points")
    else:
        split_points = np.array([0, len(list(trajectory_files.values())[0])])

    trajectory = np.array([[list(trajectory_files[key])[split_points[i]:split_points[i + 1]] for i
                            in range(len(split_points) - 1)] for key in keys], dtype=object)
    # select the first dataset of that path
    trajectory = trajectory[:, 0]

    # preprocess to get same data as function fit()
    preprocessed_traj = [list() for i in range(49)]
    xy = trajectory[:2]
    for state in trajectory.transpose():
        temp = []
        for entry in state[2:]:
            # to flatten the states (not possible with flatten() because we have as dtype=object
            if type(entry) == np.ndarray:
                temp += list(entry)
            else:
                temp.append(entry)
        obs = np.concatenate([temp,
                              np.zeros(12),
                              ]).flatten()
        new_state = obs[:34]
        # transform rotation matrix into rotation angle
        temp = np.dot(obs[34:43].reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
        angle = np.arctan2(temp[3], temp[0])
        # and turn angle to sin, cos (for a closed angle range)
        new_state = np.append(new_state, [np.cos(angle), np.sin(angle)])
        new_state = np.append(new_state, obs[43:])

        for i in range(len(new_state)):
            preprocessed_traj[i].append(new_state[i])

    preprocessed_traj = np.array(preprocessed_traj)

    # rotate_data:
    rotated_traj = rotate_obs(preprocessed_traj, rotation_angle)
    rotated_xy = [list() for x in range(2)]
    # don't needed in fit
    rotated_xy[0] = np.cos(rotation_angle) * np.array(xy[0]) - np.sin(rotation_angle) * np.array(xy[1])
    rotated_xy[1] = np.sin(rotation_angle) * np.array(xy[0]) + np.cos(rotation_angle) * np.array(xy[1])

    # post process data to be simulatable with play_trajectory_demo:
    postprocessed_traj = list(rotated_traj[:34])
    # turn sin and cos into angles again
    temp = []
    for j in range(len(rotated_traj[34])):
        temp.append(np.arccos(rotated_traj[34][j]) * (1 if np.arcsin(rotated_traj[35][j]) > 0 else -1))

    # turn angles into matrix
    postprocessed_traj.append([
        np.dot(np.array(
            [[np.cos((angle + np.pi) % (2 * np.pi) - np.pi), -np.sin((angle + np.pi) % (2 * np.pi) - np.pi), 0],
             [np.sin((angle + np.pi) % (2 * np.pi) - np.pi), np.cos((angle + np.pi) % (2 * np.pi) - np.pi), 0],
             [0, 0, 1]]),
            np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))).reshape((9,)) for angle in temp])
    postprocessed_traj.append(rotated_traj[36])

    if not os.path.exists(store_path):
        os.mkdir(store_path)

    # store new dataset
    np.savez(os.path.join(store_path, 'test_rotate_dataset_' + str(rotation_angle) + '.npz'),
             q_trunk_tx=rotated_xy[0],
             q_trunk_ty=rotated_xy[1],
             q_trunk_tz=postprocessed_traj[0],
             q_trunk_rotation=postprocessed_traj[1],
             q_trunk_list=postprocessed_traj[2],
             q_trunk_tilt=postprocessed_traj[3],
             q_FR_hip_joint=postprocessed_traj[4],
             q_FR_thigh_joint=postprocessed_traj[5],
             q_FR_calf_joint=postprocessed_traj[6],
             q_FL_hip_joint=postprocessed_traj[7],
             q_FL_thigh_joint=postprocessed_traj[8],
             q_FL_calf_joint=postprocessed_traj[9],
             q_RR_hip_joint=postprocessed_traj[10],
             q_RR_thigh_joint=postprocessed_traj[11],
             q_RR_calf_joint=postprocessed_traj[12],
             q_RL_hip_joint=postprocessed_traj[13],
             q_RL_thigh_joint=postprocessed_traj[14],
             q_RL_calf_joint=postprocessed_traj[15],
             dq_trunk_tx=postprocessed_traj[16],
             dq_trunk_ty=postprocessed_traj[17],
             dq_trunk_tz=postprocessed_traj[18],
             dq_trunk_rotation=postprocessed_traj[19],
             dq_trunk_list=postprocessed_traj[20],
             dq_trunk_tilt=postprocessed_traj[21],
             dq_FR_hip_joint=postprocessed_traj[22],
             dq_FR_thigh_joint=postprocessed_traj[23],
             dq_FR_calf_joint=postprocessed_traj[24],
             dq_FL_hip_joint=postprocessed_traj[25],
             dq_FL_thigh_joint=postprocessed_traj[26],
             dq_FL_calf_joint=postprocessed_traj[27],
             dq_RR_hip_joint=postprocessed_traj[28],
             dq_RR_thigh_joint=postprocessed_traj[29],
             dq_RR_calf_joint=postprocessed_traj[30],
             dq_RL_hip_joint=postprocessed_traj[31],
             dq_RL_thigh_joint=postprocessed_traj[32],
             dq_RL_calf_joint=postprocessed_traj[33],
             dir_arrow=postprocessed_traj[34],
             goal_speed=postprocessed_traj[35],
             split_points=[0, len(postprocessed_traj[0])])

    # and return path to first rotation dataset
    return os.path.join(store_path, 'test_rotate_dataset_' + str(rotation_angle) + '.npz')


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def interpolate_map(traj):
    """
    preprocesses the trajectory data before interpolating it:
        in this case transforming the rotation matrix into an angle
        and making sure the rotations we be interpolated right
    """
    traj_list = [list() for j in range(len(traj))]
    for i in range(len(traj_list)):
        # if the state is a rotation
        if i in [3, 4, 5]:
            # change it to the nearest rotation presentation to the previous state
            # -> no huge jumps between -pi and pi for example
            traj_list[i] = list(np.unwrap(traj[i]))
        else:
            traj_list[i] = list(traj[i])
    # turn matrix into angle
    traj_list[36] = np.unwrap([
        np.arctan2(np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                   np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
        for mat in traj[36]])
    return np.array(traj_list)


def interpolate_remap(traj):
    """
    postprocesses the trajectory data after interpolating it:
        in this case transforming an angle into the rotation matrix
        and makes sure the rotations are in the range -pi,pi
    """
    traj_list = [list() for j in range(len(traj))]
    for i in range(len(traj_list)):
        # if the state is a rotation
        if i in [3, 4, 5]:
            # make sure it is in range -pi,pi
            traj_list[i] = [(angle + np.pi) % (2 * np.pi) - np.pi for angle in traj[i]]
        else:
            traj_list[i] = list(traj[i])
    # transforms angle into rotation matrix
    traj_list[36] = [
        np.dot(np.array(
            [[np.cos((angle + np.pi) % (2 * np.pi) - np.pi), -np.sin((angle + np.pi) % (2 * np.pi) - np.pi), 0],
             [np.sin((angle + np.pi) % (2 * np.pi) - np.pi), np.cos((angle + np.pi) % (2 * np.pi) - np.pi), 0],
             [0, 0, 1]]),
               np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))).reshape((9,)) for angle in traj[36]]
    return np.array(traj_list, dtype=object)


def reward_callback(state, action, next_state):
    """
    defines the reward how we want to measure the quality of a state
        important: only a metric for the comparison of different agents
        it's the difference between the desired velocity vector and the actual velocity vector of the trunk
    """
    # actual velocity vecotr
    act_vel = np.array([state[16], state[17]])
    # desired velocity vector with desired angle/direction
    mat = np.dot(state[34:43].reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape(((9,)))
    angle = np.arctan2(mat[3], mat[0])
    norm_x = np.cos(angle)
    norm_y = np.sin(angle)

    wanted_vel = state[43] * np.array([norm_x, norm_y])
    result = act_vel - wanted_vel
    return np.exp(-np.square(np.linalg.norm(result)))


if __name__ == '__main__':

    # trajectory demo ---------------------------------------------------------------------------------------------------
    # define env and data frequencies
    # env_freq = 1000  # hz, added here as a reminder
    # traj_data_freq = 500 #500 change interpolation in test_rotate too!!! # hz, added here as a reminder
    # desired_contr_freq = 100  # hz
    # n_substeps = env_freq // desired_contr_freq
    #
    # traj_path =  '/home/moore/DataGeneration/data_generation/Quadruped_Unitree_A1/tim_quadruped_data/data/states_2023_02_23_19_48_33_straight.npz' #'/home/tim/Documents/locomotion_simulation/locomotion/examples/log/2023_02_23_19_22_49/states.npz'#
    #
    # rotation_angle = np.pi
    # # todo create dir if it does not exist
    # traj_path = test_rotate_data(traj_path, rotation_angle, store_path='./new_unitree_a1_with_dir_vec_model')
    #
    # # prepare trajectory params
    # traj_params = dict(traj_path=traj_path,
    #                    traj_dt=(1 / traj_data_freq),
    #                    control_dt=(1 / desired_contr_freq),
    #                    interpolate_map=interpolate_map, #transforms 9dim rot matrix into one rot angle
    #                    interpolate_remap=interpolate_remap # and back
    #                    )
    # gamma = 0.99
    # horizon = 1000
    #
    # env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,use_torque_ctrl=True,
    #                 traj_params=traj_params, random_start=False, init_step_no=0, init_traj_no=0,
    #                 use_2d_ctrl=True, tmp_dir_name=".",
    #                 goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))
    # action_dim = env.info.action_space.shape[0]
    # print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    # print("Dimensionality of Act-space:", env.info.action_space.shape[0])
    #
    # with catchtime() as t:
    #     env.play_trajectory_demo(desired_contr_freq)
    #     print("Time: %fs" % t())
    # exit()

    # play action demo -------------------------------------------------------------------------------------------------
    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder simulation freq
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    actions_path = '/home/moore/DataGeneration/data_generation/Quadruped_Unitree_A1/tim_quadruped_data/data/actions_position_2023_02_23_19_48_33_straight.npz'  # '/home/tim/Documents/IRL_unitreeA1/data/actions_position_2023_02_23_19_48_33_straight.npz'
    states_path = '/home/moore/DataGeneration/data_generation/Quadruped_Unitree_A1/tim_quadruped_data/data/states_2023_02_23_19_48_33_straight.npz'  # '/home/tim/Documents/IRL_unitreeA1/data/states_2023_02_23_19_48_33_straight.npz'#
    use_rendering = True
    use_pd_controller = False

    use_2d_ctrl = True
    use_torque_ctrl = False

    gamma = 0.99
    horizon = 1000

    traj_params = dict(traj_path=states_path,
                       traj_dt=1 / traj_data_freq,
                       control_dt=1 / desired_contr_freq,
                       interpolate_map=interpolate_map,  # transforms 9dim rot matrix into one rot angle
                       interpolate_remap=interpolate_remap  # and back
                       )

    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params,
                    random_start=False, init_step_no=0, init_traj_no=0,
                    use_torque_ctrl=use_torque_ctrl, use_2d_ctrl=use_2d_ctrl, tmp_dir_name=".")

    dataset = env.create_dataset()

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()

    env.play_action_demo(actions_path=actions_path,
                         use_rendering=use_rendering, use_pd_controller=use_pd_controller,
                         interpolate_map=interpolate_map, interpolate_remap=interpolate_remap)
    print("Finished")
    exit()

    # simulation demo --------------------------------------------------------------------------------------------------

    env_freq = 1000  # hz, added here as a reminder simulation freq
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    gamma = 0.99
    horizon = 1000

    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    use_torque_ctrl=False, use_2d_ctrl=True, tmp_dir_name='.', setup_random_rot=True)
    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()
    env.render()
    while True:
        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)
        env.render()




