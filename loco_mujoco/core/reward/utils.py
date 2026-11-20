"""Helper functions for reward computation."""

from typing import Union
from types import ModuleType

import numpy as np
import jax.numpy as jnp


def out_of_bounds_action_cost(action: Union[np.ndarray, jnp.ndarray],
                              lower_bound: Union[np.ndarray, jnp.ndarray],
                              upper_bound: Union[np.ndarray, jnp.ndarray],
                              backend: ModuleType,
                              cost_fn: str = "squared"):
    """
    Calculate the cost of an action that is out of bounds.

    Args:
        action (Union[np.ndarray, jnp.ndarray]): The action to be evaluated.
        lower_bound (Union[np.ndarray, jnp.ndarray]): The lower bound of the action space.
        upper_bound (Union[np.ndarray, jnp.ndarray]): The upper bound of the action space.
        backend (ModuleType): The backend module used for calculation (e.g., numpy or jax.numpy).
        cost_fn (str): The cost function to be used. Can be either "squared" or "abs".

    Returns:
        float: The cost of the action.

    """

    action_dim = lower_bound.shape[0]

    lower_cost = backend.where(action < lower_bound, (lower_bound - action), 0)
    upper_cost = backend.where(action > upper_bound, (action - upper_bound), 0)

    if cost_fn == "squared":
        cost_fn = backend.square
    elif cost_fn == "abs":
        cost_fn = backend.abs
    else:
        raise ValueError("Invalid cost function: {}".format(cost_fn))

    return backend.sum(cost_fn(lower_cost + upper_cost)) / action_dim
