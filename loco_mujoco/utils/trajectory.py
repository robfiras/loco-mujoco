import warnings
from copy import deepcopy

import numpy as np
from scipy import interpolate


class Trajectory(object):
    """
    General class to handle trajectory data. It builds a general trajectory from a numpy bin file(.npy), and
    automatically interpolates the trajectory to the desired control frequency. This class is used to generate datasets
    and to sample from the dataset to initialize the simulation.

    All trajectories are required to be of equal length.

    """
    def __init__(self, keys, traj_path, low, high, joint_pos_idx, traj_dt=0.002, control_dt=0.01, ignore_keys=None,
                 interpolate_map=None, interpolate_remap=None):
        """
        Constructor.

        Args:
            keys (list): List of keys to extract data from the trajectories.
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations).
            traj_dt (float): Time step of the trajectory file.
            control_dt (float): Model control frequency used to interpolate the trajectory.
            ignore_keys (list): List of keys to ignore in the dataset.
            interpolate_map (func): Function used to map a trajectory to some space that allows interpolation.
            interpolate_remap (func): Function used to map a transformed trajectory back to the original space after
                interpolation.

        """

        # load data
        self._trajectory_files = np.load(traj_path, allow_pickle=True)

        # convert to dict to be mutable
        self._trajectory_files = {k: d for k, d in self._trajectory_files.items()}
        self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx)

        # add all goals to keys (goals have to start with 'goal' if not in keys)
        keys += [key for key in self._trajectory_files.keys() if key.startswith('goal') and key not in keys]
        self.keys = keys

        # remove unwanted keys
        if ignore_keys is not None:
            for ik in ignore_keys:
                keys.remove(ik)

        # split points mark the beginning of the next trajectory.
        # The last split_point points to the index behind the last element of the trajectories -> len(traj)
        if "split_points" in self._trajectory_files.keys():
            self.split_points = self._trajectory_files["split_points"]
        else:
            self.split_points = np.array([0, len(list(self._trajectory_files.values())[0])])

        # 3d matrix: (len state space, no trajectories, no of samples)
        self.trajectories = np.array([[list(self._trajectory_files[key])[self.split_points[i]:self.split_points[i+1]]
                                     for i in range(len(self.split_points)-1)] for key in keys], dtype=object)

        self.traj_dt = traj_dt
        self.control_dt = control_dt

        # interpolation of the trajectories
        if self.traj_dt != control_dt:
            self._interpolate_trajectories(map_funct=interpolate_map, re_map_funct=interpolate_remap)

        self.subtraj_step_no = 0
        self.traj_no = 0
        self.subtraj = self.trajectories[:, self.traj_no].copy()

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset used by imitation learning algorithms.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj).

        """
        flat_traj = self.flattened_trajectories()

        # create a dict and extract all elements except the ones specified in ignore_keys.
        all_data = dict(zip(self.keys, deepcopy(list(flat_traj))))
        if ignore_keys is not None:
            for ikey in ignore_keys:
                del all_data[ikey]

        traj = list(all_data.values())
        states = np.transpose(deepcopy(np.array(traj)))

        # convert to dict with states and next_states
        new_states = states[:-1]
        new_next_states = states[1:]
        absorbing = np.zeros(len(states[:-1]))  # we assume that there are no absorbing states in the trajectory

        return dict(states=new_states, next_states=new_next_states, absorbing=absorbing)

    def _interpolate_trajectories(self, map_funct=None, re_map_funct=None):
        """
        Interpolates all trajectories.

        Args:
            map_funct (func): Function used to map a trajectory to some space that allows interpolation.
            re_map_funct (func): Function used to map a transformed trajectory back to the original space after
                interpolation.

        """
        assert (map_funct is None) == (re_map_funct is None)

        new_trajs = list()

        # interpolate over each trajectory
        for traj in np.rollaxis(self.trajectories, 1):

            x = np.arange(traj.shape[1])
            new_traj_sampling_factor = self.traj_dt / self.control_dt
            x_new = np.linspace(0, traj.shape[1] - 1, round(traj.shape[1] * new_traj_sampling_factor), endpoint=True)

            # preprocess trajectory
            if map_funct is not None:
                traj = map_funct(traj)

            new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)

            # postprocess trajectory
            if re_map_funct is not None:
                new_traj = re_map_funct(new_traj)

            new_trajs.append(new_traj)

        self.trajectories = np.concatenate(new_trajs, axis=1)

        # interpolation of split_points
        self.split_points = [0]
        for k in range(self.number_of_trajectories):
            self.split_points.append(self.split_points[-1] + len(self.trajectories[0][k]))

    def reset_trajectory(self, substep_no=None, traj_no=None):
        """
        Resets the trajectory to a certain trajectory and a substep within that trajectory. If one of them is None,
        they are set randomly.

        Args:
            substep_no (int, None): Starting point of the trajectory.
                If None, the trajectory starts from a random point.
            traj_no (int, None): Number of the trajectory to start from.
                If None, it starts from a random trajectory

        Returns:
            The chosen (or randomly sampled) sample from a trajectory.

        """

        if traj_no is None:
            self.traj_no = np.random.randint(0, self.number_of_trajectories)
        else:
            assert 0 <= traj_no <= self.number_of_trajectories
            self.traj_no = traj_no

        if substep_no is None:
            self.subtraj_step_no = np.random.randint(0, self.trajectory_length)
        else:
            assert 0 <= substep_no <= self.trajectory_length
            self.subtraj_step_no = substep_no

        # choose a sub trajectory
        self.subtraj = self.trajectories[:, self.traj_no].copy()

        # reset x and y to middle position
        self.subtraj[0] -= self.subtraj[0, self.subtraj_step_no]
        self.subtraj[1] -= self.subtraj[1, self.subtraj_step_no]

        return self.subtraj[:, self.subtraj_step_no]

    def check_if_trajectory_is_in_range(self, low, high, keys, j_idx):

        # get q_pos indices
        j_idx = j_idx[2:]   # exclude x and y

        # check if they are in range
        for i, item in enumerate(self._trajectory_files.items()):
            k, d = item
            if i in j_idx:
                high_i = high[i-2]
                low_i = low[i-2]
                if np.max(d) > high_i:
                    warnings.warn("Trajectory violates joint range in %s. Maximum in trajectory is %f "
                                  "and maximum range is %f. Clipping the trajectory into range!"
                                  % (keys[i], np.max(d), high_i), RuntimeWarning)
                elif np.min(d) < low_i:
                    warnings.warn("Trajectory violates joint range in %s. Minimum in trajectory is %f "
                                  "and minimum range is %f. Clipping the trajectory into range!"
                                  % (keys[i], np.min(d), low_i), RuntimeWarning)

                # clip trajectory to min & max
                self._trajectory_files[k] = np.clip(self._trajectory_files[k], low_i, high_i)

    def get_next_sample(self):
        """
        Returns the next sample in the trajectory.

        """
        self.subtraj_step_no += 1
        if self.subtraj_step_no == self.trajectory_length:
            sample = self.reset_trajectory(substep_no=0)
        else:
            sample = deepcopy(self.subtraj[:, self.subtraj_step_no])

        return sample

    def flattened_trajectories(self):
        """
        Returns the trajectories flattened in the N_traj dimension.

        """
        return self.trajectories.reshape((len(self.trajectories), -1))

    @property
    def trajectory_length(self):
        """
        Returns the length of a trajectory. Note that all trajectories have to be equal in length.

        """
        return self.trajectories.shape[-1]

    @property
    def number_of_trajectories(self):
        """
        Returns the number of trajectories.

        """
        return self.trajectories.shape[1]
