import warnings
from copy import deepcopy

import numpy as np
from scipy import interpolate


class Trajectory:
    """
    General class to handle trajectory data. It builds a general trajectory from a numpy bin file(.npy), and
    automatically interpolates the trajectory to the desired control frequency. This class is used to generate datasets
    and to sample from the dataset to initialize the simulation.

    All trajectories are required to be of equal length.

    """
    def __init__(self, keys, low, high, joint_pos_idx, interpolate_map, interpolate_remap,
                 traj_path=None, traj_files=None, interpolate_map_params=None, interpolate_remap_params=None,
                 traj_dt=0.002, control_dt=0.01, ignore_keys=None, clip_trajectory_to_joint_ranges=False,
                 traj_info=None, warn=True):
        """
        Constructor.

        Args:
            keys (list): List of keys to extract data from the trajectories.
            low (np.array): Lower bound of the trajectory values.
            high (np.array): Upper bound of the trajectory values.
            joint_pos_idx (np.array): Array including all indices of the joint positions in the trajectory.
            interpolate_map (func): Function used to map a trajectory to some space that allows interpolation.
            interpolate_remap (func): Function used to map a transformed trajectory back to the original space after
                interpolation.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a 'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            traj_files (dict): Dictionary containing all trajectory files. If traj_path is specified, this
                should be None.
            interpolate_map_params: Set of parameters needed to do the interpolation by the Unitree environment.
            interpolate_remap_params: Set of parameters needed to do the interpolation by the Unitree environment.
            traj_dt (float): Time step of the trajectory file.
            control_dt (float): Model control frequency used to interpolate the trajectory.
            ignore_keys (list): List of keys to ignore in the dataset.
            clip_trajectory_to_joint_ranges (bool): If True, the joint positions in the trajectory are clipped
                between the low and high values in the trajectory.
            traj_info (list): A list of custom labels for each trajectory.
            warn (bool): If True, a warning will be raised, if some trajectory ranges are violated.

        """

        assert (traj_path is not None) != (traj_files is not None), "Please specify either traj_path or " \
                                                                    "traj_files, but not both."

        # load data
        if traj_path is not None:
            self._trajectory_files = np.load(traj_path, allow_pickle=True)
        else:
            self._trajectory_files = traj_files

        # convert to dict to be mutable
        self._trajectory_files = {k: d for k, d in self._trajectory_files.items()}

        self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx, warn, clip_trajectory_to_joint_ranges)

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

        #  Extract trajectory from files. This returns a list of np.arrays. The length of the
        #  list is the number of observations. Each np.array has the shape
        #  (n_trajectories, n_samples, (dim_observation)). If dim_observation is one
        #  the shape of the array is just (n_trajectories, n_samples).
        self.trajectories = self._extract_trajectory_from_files()

        if traj_info is not None:
            assert len(traj_info) == self.number_of_trajectories, "The number of trajectory infos/labels need " \
                                                                  "to be equal to the number of trajectories."
        self._traj_info = traj_info

        self.traj_dt = traj_dt
        self.control_dt = control_dt

        # interpolation of the trajectories
        if self.traj_dt != control_dt:
            self._interpolate_trajectories(map_funct=interpolate_map,
                                           map_params=interpolate_map_params,
                                           re_map_funct=interpolate_remap,
                                           re_map_params=interpolate_remap_params)

        self.subtraj_step_no = 0
        self.traj_no = 0
        self.subtraj = self._get_subtraj(self.traj_no)

    def create_dataset(self, ignore_keys=None, state_callback=None, state_callback_params=None):
        """
        Creates a dataset used by imitation learning algorithms.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.
            state_callback (func): Function that should be called on each state.
            state_callback_params (dict): Dictionary of parameters needed to make
                the state transformation.

        Returns:
            Dictionary containing states, next_states, absorbing and last flags. For the states the shape is
            (N_traj x N_samples_per_traj-1, dim_state), while the flags have the shape
            (N_traj x N_samples_per_traj-1). If traj_info was specified, it will also include that.

        """
        flat_traj = self.flattened_trajectories()

        # create a dict and extract all elements except the ones specified in ignore_keys.
        all_data = dict(zip(self.keys, deepcopy(list(flat_traj))))
        if ignore_keys is not None:
            for ikey in ignore_keys:
                del all_data[ikey]

        traj = list(all_data.values())

        # create states array shape=(n_states, dim_obs)
        states = np.concatenate(traj, axis=1)

        if state_callback is not None:
            transformed_states = []
            for state in states:
                transformed_states.append(state_callback(state, **state_callback_params))
            states = np.array(transformed_states)

        # convert to dict with states and next_states
        new_states = states[:-1]
        new_next_states = states[1:]
        absorbing = np.zeros(len(states[:-1]))  # we assume that there are no absorbing states in the trajectory
        last = np.zeros(len(states))
        last[self.split_points[1:]-1] = np.ones(len(self.split_points)-1)

        if self._traj_info is not None:
            info = np.array([[l] * self.trajectory_length for l in self._traj_info]).reshape(-1)
            return dict(states=new_states, next_states=new_next_states, absorbing=absorbing, last=last, info=info)
        else:
            return dict(states=new_states, next_states=new_next_states, absorbing=absorbing, last=last)

    def _extract_trajectory_from_files(self):
        """
        Extracts the trajectory from the trajectory files by filtering for the relevant keys.
        The trajectory is then split to multiple trajectories using the split points.

        Returns:
            A list of np.arrays. The length of the list is the number of observations.
            Each np.array has the shape (n_trajectories, n_samples, (dim_observation)).
            If dim_observation is one the shape of the array is just (n_trajectories, n_samples).

        """

        # load data of relevant keys
        trajectories = [self._trajectory_files[key] for key in self.keys]

        # check that all observations have equal lengths
        len_obs = np.array([len(obs) for obs in trajectories])
        assert np.all(len_obs == len_obs[0]), "Some observations have different lengths than others. " \
                                              "Trajectory is corrupted. "

        # split trajectory into multiple trajectories using split points
        for i in range(len(trajectories)):
            trajectories[i] = np.split(trajectories[i], self.split_points[1:-1])
            # check if all trajectories are of equal length
            len_trajectories = np.array([len(traj) for traj in trajectories[i]])
            assert np.all(len_trajectories == len_trajectories[0]), "Only trajectories of equal length " \
                                                                    "are currently supported."
            trajectories[i] = np.array(trajectories[i])

        return trajectories

    def _interpolate_trajectories(self, map_funct, re_map_funct, map_params, re_map_params):
        """
        Interpolates all trajectories cubically.

        Args:
            map_funct (func): Function used to map a trajectory to some space that allows interpolation.
            re_map_funct (func): Function used to map a transformed trajectory back to the original space after
                interpolation.
            map_params: Set of parameters needed to do the interpolation by the respective environment.
            re_map_params: Set of parameters needed to do the interpolation by the respective environment.

        """

        assert (map_funct is None) == (re_map_funct is None)

        new_trajs = list()

        # interpolate over each trajectory
        for i in range(self.number_of_trajectories):

            traj = [obs[i] for obs in self.trajectories]

            x = np.arange(self.trajectory_length)
            new_traj_sampling_factor = self.traj_dt / self.control_dt
            x_new = np.linspace(0, self.trajectory_length - 1, round(self.trajectory_length * new_traj_sampling_factor),
                                endpoint=True)

            # preprocess trajectory
            traj = map_funct(traj) if map_params is None else map_funct(traj, **map_params)

            new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)

            # postprocess trajectory
            new_traj = re_map_funct(new_traj) if re_map_params is None else re_map_funct(new_traj, **re_map_params)

            new_trajs.append(new_traj)

        # convert trajectory back to original shape
        trajectories = []
        for i in range(self.number_obs_trajectory):
            trajectories.append([])
            for traj in new_trajs:
                trajectories[i].append(traj[i])
            trajectories[i] = np.array(trajectories[i])
        self.trajectories = trajectories

        # interpolation of split_points
        self.split_points = [0]
        for k in range(self.number_of_trajectories):
           self.split_points.append(self.split_points[-1] + len(self.trajectories[0][k]))
        self.split_points = np.array(self.split_points)

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
        self.subtraj = self._get_subtraj(self.traj_no)

        # reset x and y to middle position
        self.subtraj[0] -= self.subtraj[0][self.subtraj_step_no]
        self.subtraj[1] -= self.subtraj[1][self.subtraj_step_no]

        sample = [obs[self.subtraj_step_no] for obs in self.subtraj]

        return sample

    def check_if_trajectory_is_in_range(self, low, high, keys, j_idx, warn, clip_trajectory_to_joint_ranges):

        if warn or clip_trajectory_to_joint_ranges:

            # get q_pos indices
            j_idx = j_idx[2:]   # exclude x and y
            highs = dict(zip(keys[2:], high))
            lows = dict(zip(keys[2:], low))

            # check if they are in range
            for i, item in enumerate(self._trajectory_files.items()):
                k, d = item
                if i in j_idx and k in keys:
                    if warn:
                        clip_message = "Clipping the trajectory into range!" if clip_trajectory_to_joint_ranges else ""
                        if np.max(d) > highs[k]:
                            warnings.warn("Trajectory violates joint range in %s. Maximum in trajectory is %f "
                                          "and maximum range is %f. %s"
                                          % (k, np.max(d), highs[k], clip_message), RuntimeWarning)
                        elif np.min(d) < lows[k]:
                            warnings.warn("Trajectory violates joint range in %s. Minimum in trajectory is %f "
                                          "and minimum range is %f. %s"
                                          % (k, np.min(d), lows[k], clip_message), RuntimeWarning)

                    # clip trajectory to min & max
                    if clip_trajectory_to_joint_ranges:
                        self._trajectory_files[k] = np.clip(self._trajectory_files[k], lows[k], highs[k])

    def get_current_sample(self):
        """
        Returns the current sample in the trajectory.

        """

        return self._get_ith_sample_from_subtraj(self.subtraj_step_no)

    def get_next_sample(self):
        """
        Returns the next sample in the trajectory.

        """

        self.subtraj_step_no += 1
        if self.subtraj_step_no == self.trajectory_length:
            sample = None
        else:
            sample = self._get_ith_sample_from_subtraj(self.subtraj_step_no)

        return sample

    def get_from_sample(self, sample, key):
        """
        Returns the part of the sample whose key is specified. In contrast to the
        function _get_from_obs from the base environment, this function also allows to
        access information that is in the trajectory, but not in the simulation such
        as goal definitions.

        Note: This function is not suited for getting an observation from environment samples!

        Args:
            sample (list or np.array): Current sample to extract an observation from.
            key (string): Name of the observation to extract from sample

        Returns:
            np.array consisting of the observation specified by the key.

        """
        assert len(sample) == len(self.keys)

        idx = self.get_idx(key)

        return sample[idx]

    def get_idx(self, key):
        """
        Returns the index of the key.

        Note: This function is not suited for getting the index for an observation of the environment!

        Args:
            key (string): Name of the observation to extract from sample

        Returns:
            int containing the desired index.

        """

        return self.keys.index(key)

    def flattened_trajectories(self):
        """
        Returns the trajectories flattened in the N_traj dimension. Also expands dim if obs has dimension 1.

        """
        trajectories = []
        for obs in self.trajectories:
            if len(obs.shape) == 2:
                trajectories.append(obs.reshape((-1, 1)))
            elif len(obs.shape) == 3:
                trajectories.append(obs.reshape((-1, obs.shape[2])))
            else:
                raise ValueError("Unsupported shape of observation %s." % obs.shape)

        return trajectories

    def _get_subtraj(self, i):
        """
        Returns a copy of the i-th trajectory included in trajectories.

        """

        return [obs[i].copy() for obs in self.trajectories]

    def _get_ith_sample_from_subtraj(self, i):
        """
        Returns a copy of the i-th sample included in the current subtraj.

        """

        return [np.array(obs[i].copy()).flatten() for obs in self.subtraj]

    @property
    def number_obs_trajectory(self):
        """
        Returns the number of observations in the trajectory.

        """
        return len(self.trajectories)

    @property
    def trajectory_length(self):
        """
        Returns the length of a trajectory. Note that all trajectories have to be equal in length.

        """
        return self.trajectories[0].shape[1]

    @property
    def number_of_trajectories(self):
        """
        Returns the number of trajectories.

        """
        return self.trajectories[0].shape[0]
