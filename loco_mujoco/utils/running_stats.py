"""
Running statistics with a functional interface for use in JAX code.

The statistics are kept in immutable state dataclasses, which are created by ``reset`` and replaced by
``update_state``.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStandardizationState:
    """
    State of :class:`RunningStandardization`.

    Attributes:
        mean (jnp.array): Running mean.
        std (jnp.array): Running spread statistic accumulated by the update rule of Welford's algorithm.
        count (int): Number of samples seen so far.

    """
    mean: jnp.array
    std: jnp.array
    count: int


@struct.dataclass
class RunningAverageWindowState:
    """
    State of :class:`RunningAveragedWindow`.

    Attributes:
        storage (jnp.array): Ring buffer with the last values, shape ``(window_size, *shape)``.
        index (int): Position in the buffer at which the next value is written.
        curr_size (int): Number of values currently stored, at most ``window_size``.

    """
    storage: jnp.array
    index: int = 0
    curr_size: int = 0


class RunningStandardization:
    """
    Compute running statistics of values for standardization, following Welford's online algorithm:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    The statistics are stored in a :class:`RunningStandardizationState`.

    """
    def __init__(self, shape, alpha=1e-32):
        """
        Constructor.

        Args:
            shape (tuple): Shape of the data to standardize.
            alpha (float): Minimum learning rate; has to be in the open interval ``(0, 1)``.

        """
        self._shape = shape

        assert 0. < alpha < 1.
        self._alpha = alpha

    def reset(self):
        """
        Initialize the running standardization state.

        Returns:
            RunningStandardizationState: State with zero mean, unit spread and a count of one.

        """

        running_std_state = RunningStandardizationState(mean=jnp.zeros(1, *self._shape),
                                                        std=jnp.ones(1, *self._shape),
                                                        count=1)
        return running_std_state

    def update_state(self, value, state: RunningStandardizationState):
        """
        Update the statistics with the current data value.

        Args:
            value (jnp.array): Current data value or batch of values, shape ``shape`` or ``(batch_size, *shape)``.
            state (RunningStandardizationState): Current state of the running standardization.

        Returns:
            RunningStandardizationState: State with updated statistics.

        """

        value = jnp.atleast_2d(value)
        batch_size = len(value)

        new_c = state.count + batch_size
        alpha = max(batch_size / state.count, self._alpha)
        new_m = (1 - alpha) * state.mean + alpha * value.mean(0)
        new_s = state.std + (value.mean(0) - state.mean) * (value.mean(0) - new_m)
        state = state.replace(mean=new_m, std=new_s, count=new_c)

        return state


class RunningAveragedWindow:
    """
    Compute the running average using a window of fixed size.

    """
    def __init__(self, shape, window_size):
        """
        Constructor.

        Args:
            shape (tuple): Shape of the data to average.
            window_size (int): Number of values the window holds.

        """
        self._shape = shape
        self._window_size = window_size

    def reset(self):
        """
        Initialize the running average window state.

        Returns:
            RunningAverageWindowState: State with an empty window.

        """
        state = RunningAverageWindowState(storage=jnp.zeros(self._window_size, *self._shape),
                                          index=0,
                                          curr_size=0)

        return state

    def update_state(self, value, state: RunningAverageWindowState):
        """
        Add the current data value to the window, overwriting the oldest value once the window is full.

        Args:
            value (jnp.ndarray): Current data value to use for the update.
            state (RunningAverageWindowState): Current state of the running average window.

        Returns:
            RunningAverageWindowState: The updated state.

        """
        value = jnp.atleast_2d(value)
        new_storage = state.storage.at[state.index].set(value)
        new_index = jax.lax.cond(state.index+1 >= self._window_size,
                                 lambda: 0, lambda: state.index+1)
        new_size = jax.lax.cond(state.curr_size >= self._window_size,
                                lambda: state.curr_size, lambda: state.curr_size + 1)
        return state.replace(storage=new_storage, index=new_index, curr_size=new_size)

    @staticmethod
    def mean(state):
        """
        Compute the mean of the stored values.

        Args:
            state (RunningAverageWindowState): Current state of the running average window.

        Returns:
            jnp.ndarray: Mean of the values currently stored in the window.

        """
        mean = state.storage.sum(0)
        return mean / state.curr_size
