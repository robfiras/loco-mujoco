import numpy as np

from loco_mujoco import LocoEnv

from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box


class GymnasiumWrapper(Env):
    """
    This class implements a simple wrapper to use all LocoMuJoCo environments
    with the Gymnasium interface.

    """

    def __init__(self, env_name, **kwargs):
        self.spec = EnvSpec(env_name)

        self._env = LocoEnv.make(env_name, **kwargs)

        self.observation_space = self._convert_space(self._env.info.observation_space)
        self.action_space = self._convert_space(self._env.info.action_space)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        .. note:: We set the truncation always to False, as the time limit can be handled by the `TimeLimit`
         wrapper in Gymnasium.

        Args:
            action (np.ndarray):
                The action to be executed in the environment.

        Returns:
            Tuple of observation, reward, terminated, truncated and info.

        """

        obs, reward, absorbing, info = self._env.step(action)

        return obs, reward, absorbing, False, info

    def reset(self, *, seed=None, options=None):
        """
        Resets the state of the environment, returning an initial observation and info.

        """

        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self._env.reset(), {}

    def render(self):
        """
        Renders the environment.

        """
        return self._env.render()

    def close(self):
        """
        Closes the environment.

        """
        self._env.stop()

    def create_dataset(self, **kwargs):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset.

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj). For perfect and preference datasets, the actions are also provided.

        """
        return self._env.create_dataset(**kwargs)

    def play_trajectory(self, **kwargs):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the trajectories at every step.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the rendered trajectory will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """
        return self._env.play_trajectory(**kwargs)

    def play_trajectory_from_velocity(self, **kwargs):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones calculated from the joint velocities
        in the trajectories at every step. Therefore, the joint positions
        are set from the trajectory in the first step. Afterwards, numerical
        integration is used to calculate the next joint positions using
        the joint velocities in the trajectory.

        Args:
            n_episodes (int): Number of episode to replay.
            n_steps_per_episode (int): Number of steps to replay per episode.
            render (bool): If True, trajectory will be rendered.
            record (bool): If True, the replay will be recorded.
            recorder_params (dict): Dictionary containing the recorder parameters.

        """
        return self._env.play_trajectory_from_velocity(**kwargs)

    @property
    def unwrapped(self):
        """
        Returns the inner environment.
        """
        return self._env

    @staticmethod
    def _convert_space(space):
        """ Converts the observation and action space from mushroom-rl to gymnasium. """
        low = np.min(space.low)
        high = np.max(space.high)
        shape = space.shape
        return Box(low, high, shape, np.float64)
