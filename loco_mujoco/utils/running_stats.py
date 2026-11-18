import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStandardizationState:
    mean: jnp.array
    std: jnp.array
    count: int


@struct.dataclass
class RunningAverageWindowState:
    storage: jnp.array
    index: int = 0
    curr_size: int = 0


class RunningStandardization:
    """
    Compute a running standardization of values according to Welford's online
    algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    def __init__(self, shape, alpha=1e-32):
        """
        Constructor.

        Args:
            shape (tuple): shape of the data to standardize;
            alpha (float, 1e-32): minimum learning rate.

        """
        self._shape = shape

        assert 0. < alpha < 1.
        self._alpha = alpha

    def reset(self):
        """
        Initialize the running standardization state.

        """

        running_std_state = RunningStandardizationState(mean=jnp.zeros((1, *self._shape)),
                                                        std=jnp.ones((1, *self._shape)),
                                                        count=1)
        return running_std_state

    def update_state(self, value, state: RunningStandardizationState):
        """
        Update the statistics with the current data value.

        Args:
            value (jnp.array): current data value to use for the update.
            state (RunningStandardizationState): current state of the running standardization.

        Returns:
            RunningStandardizationState with updated statistics.

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
            shape (tuple): shape of the data to standardize;
            window_size (int): size of the windows;

        """
        self._shape = shape
        self._window_size = window_size

    def reset(self):
        """
        Initialize the running average window state.

        """
        state = RunningAverageWindowState(storage=jnp.zeros((self._window_size, *self._shape)),
                                          index=0,
                                          curr_size=0)

        return state

    def update_state(self, value, state: RunningAverageWindowState):
        """
        Update the statistics with the current data value.

        Args:
            value (jnp.ndarray): current data value to use for the update.
            state (RunningAverageWindowState): current state of the running average window.

        """
        value = jnp.reshape(jnp.asarray(value), self._shape)
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
            state (RunningAverageWindowState): current state of the running average window.

        Returns:
            jnp.ndarray with the mean of the stored values.

        """
        mean = state.storage.sum(0)
        return mean / state.curr_size
