import os.path
import warnings
from pathlib import Path
from copy import deepcopy

from mushroom_rl.utils.running_stats import *

import loco_mujoco
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import check_validity_task_mode_dataset


class BaseRobotHumanoid(LocoEnv):
    """
    Base Class for the Mujoco simulation of Atlas and Talos.

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "carry"],
                                     data_types=["real"])

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

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        low, high = super(BaseRobotHumanoid, self)._get_observation_space()
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

        obs = super(BaseRobotHumanoid, self)._create_observation(obs)
        if self._hold_weight:
            weight_mass = deepcopy(self._model.body("weight").mass)
            obs = np.concatenate([obs, weight_mass])

        return obs

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
    def generate(env, path, task="walk", dataset_type="real", gamma=0.99, horizon=1000, random_env_reset=True, disable_arms=True,
                 disable_back_joint=False, use_foot_forces=False, random_start=True, init_step_no=None,
                 debug=False, hide_menu_on_startup=False, use_absorbing_states=True):
        """
        Returns an environment corresponding to the specified task.

        Args:
            env (class): Humanoid class, either HumanoidTorque or HumanoidMuscle.
            path (str): Path to the dataset.
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
            use_absorbing_states (bool): If True, absorbing states are defined for each environment. This means
                that episodes can terminate earlier.

        Returns:
            An MDP of the Robot.

        """
        check_validity_task_mode_dataset(BaseRobotHumanoid.__name__, task, None, dataset_type,
                                         *BaseRobotHumanoid.valid_task_confs.get_all())

        reward_params = dict(target_velocity=1.25)

        # Generate the MDP
        if task == "walk":
            mdp = env(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                      disable_arms=disable_arms, disable_back_joint=disable_back_joint,
                      use_foot_forces=use_foot_forces, reward_type="target_velocity",
                      reward_params=reward_params, random_env_reset=random_env_reset,
                      hide_menu_on_startup=hide_menu_on_startup, use_absorbing_states=use_absorbing_states)
        elif task == "carry":
            mdp = env(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                      disable_arms=disable_arms, disable_back_joint=disable_back_joint,
                      use_foot_forces=use_foot_forces, hold_weight=True, reward_type="target_velocity",
                      reward_params=reward_params, random_env_reset=random_env_reset,
                      hide_menu_on_startup=hide_menu_on_startup, use_absorbing_states=use_absorbing_states)

        # Load the trajectory
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
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
                               control_dt=(1 / desired_contr_freq))
        elif dataset_type == "perfect":
            # todo: generate and add this dataset
            raise ValueError(f"currently not implemented.")

        mdp.load_trajectory(traj_params, warn=False)

        return mdp



