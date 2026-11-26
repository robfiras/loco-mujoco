"""Checks for the array backend (NumPy or JAX) of the dual-backend functions."""

from types import ModuleType

import numpy as np
import jax.numpy as jnp


def assert_backend_is_supported(module: ModuleType):
    """
    Check if the given module is supported.

    Args:
        module (ModuleType): The module to check (e.g., numpy or jax.numpy).

    Raises:
        AssertionError: If the module is neither ``numpy`` nor ``jax.numpy``.

    """
    is_supporter = module in {np, jnp}
    assert is_supporter, f"Unsupported backend module: {module.__name__}"

