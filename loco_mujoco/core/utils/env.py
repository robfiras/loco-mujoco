"""Description of a Markov decision process: the box space and the ``MDPInfo`` container."""

import numpy as np


class Box:
    """
    This class implements functions to manage continuous states and action
    spaces. It is similar to the ``Box`` class in ``gym.spaces.box``.

    """
    def __init__(self, low, high, shape=None):
        """
        Constructor.

        Args:
            low ([float, np.ndarray]): the minimum value of each dimension of
                the space. If a scalar value is provided, this value is
                considered as the minimum one for each dimension. If a
                np.ndarray is provided, each i-th element is considered the
                minimum value of the i-th dimension;
            high ([float, np.ndarray]): the maximum value of dimensions of the
                space. If a scalar value is provided, this value is considered
                as the maximum one for each dimension. If a np.ndarray is
                provided, each i-th element is considered the maximum value
                of the i-th dimension;
            shape (np.ndarray, None): the dimension of the space. Must match
                the shape of ``low`` and ``high``, if they are np.ndarray.

        """
        if shape is None:
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = low
            self._high = high
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

    @property
    def low(self):
        """
        Returns:
             The minimum value of each dimension of the space.

        """
        return self._low

    @property
    def high(self):
        """
        Returns:
             The maximum value of each dimension of the space.

        """
        return self._high

    @property
    def shape(self):
        """
        Returns:
            The dimensions of the space.

        """
        return self._shape


class MDPInfo:
    """
    This class is used to store the information of the environment.

    """
    def __init__(self, observation_space, action_space, gamma, horizon, dt=1e-1):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space.
             action_space ([Box, Discrete]): the action space.
             gamma (float): the discount factor.
             horizon (int): the horizon.
             dt (float, 1e-1): the control timestep of the environment.

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon
        self.dt = dt

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only works for discrete spaces.

        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action spaces.

        """
        return self.observation_space.shape + self.action_space.shape
