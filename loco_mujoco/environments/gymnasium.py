import numpy as np

from loco_mujoco import LocoEnv

from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box


class GymnasiumWrapper(Env):
    """
    This class implements a simple wrapper to use all LocoMujoco environments
    with the Gymnasium interface.

    """

    def __init__(self, env_name, **kwargs):
        self.spec = EnvSpec(env_name)

        self._env = LocoEnv.make(env_name, **kwargs)

        self.observation_space = self._convert_space(self._env.info.observation_space)
        self.action_space = self._convert_space(self._env.info.action_space)

    def step(self, action):

        obs, reward, terminated, info = self._env.step(action)

        return obs, reward, terminated, False, info

    def reset(self, *, seed=None, options=None):

        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self._env.reset(), {}

    def render(self):
        return self._env.render()

    def close(self):
        self._env.stop()

    def create_dataset(self, **kwargs):
        return self._env.create_dataset(**kwargs)

    def play_trajectory(self, **kwargs):
        return self._env.play_trajectory(**kwargs)

    def play_trajectory_from_velocity(self, **kwargs):
        return self._env.play_trajectory_from_velocity(**kwargs)

    @property
    def unwrapped(self):
        return self._env

    @staticmethod
    def _convert_space(space):
        """ Converts the observation and action space from mushroom-rl to gymnasium. """
        low = np.min(space.low)
        high = np.max(space.high)
        shape = space.shape
        return Box(low, high, shape, np.float64)
