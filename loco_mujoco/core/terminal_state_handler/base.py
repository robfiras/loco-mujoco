from typing import Dict, Any, Optional, Union, Tuple
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class TerminalStateHandler(StatefulObject):
    """
    Base interface for all terminal state handlers.
    """

    registered: Dict[str, type] = dict()

    def __init__(self, env: Any, **handler_config: Dict[str, Any]):
        """
        Initialize the TerminalStateHandler.

        Args:
            env (Any): The environment instance.
            **handler_config (Any): Configuration dictionary.
        """
        self._info_props = env._get_all_info_properties()
        self._handler_config = handler_config

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terminal state handler.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def is_absorbing(self,
                     env: Any,
                     obs: np.ndarray,
                     info: Dict[str, Any],
                     data: MjData,
                     carry: Any,
                     traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Function for CPU Mujoco.

        Args:
            env (Any): The environment instance.
            obs (np.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (MjData): The Mujoco data structure.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def mjx_is_absorbing(self,
                         env: Any,
                         obs: jnp.ndarray,
                         info: Dict[str, Any],
                         data: Data,
                         carry: Any,
                         traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Function for Mjx.

        Args:
            env (Any): The environment instance.
            obs (jnp.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Data): The Mujoco data structure for Mjx.
            carry (Any): Additional carry information.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def mjx_absorbing_prob(self,
                           env: Any,
                           obs: jnp.ndarray,
                           info: Dict[str, Any],
                           data: Data,
                           carry: Any,
                           traj=None) -> Optional[jnp.ndarray]:
        """
        Smooth, differentiable version of mjx_is_absorbing, used by algorithms that backpropagate through
        imagined rollouts. Handlers without one return None, and such algorithms fall back to a learned estimate.

        Handlers implement it as 1 - prod(1 - sigmoid(excess / temperature)) over the same excesses (error minus
        threshold) that their hard check compares with zero, see ``soft_any``. Self-collision conditions have no
        smooth version: imagination computes no contacts.

        Args:
            env (Any): The environment instance.
            obs (jnp.ndarray): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Data): The Mujoco data structure for Mjx.
            carry (Any): Additional carry information.
            traj: Trajectory to use (optional).

        Returns:
            Optional[jnp.ndarray]: Probability in [0, 1] that the current state is terminal, or None if the
            handler has no smooth version.
        """
        return None

    @staticmethod
    def soft_any(violations) -> jnp.ndarray:
        """
        Smooth version of ``any(excess > 0)``: 1 - prod(1 - sigmoid(excess / temperature)).

        Args:
            violations: Iterable of (excess, temperature) pairs; the excess is positive past the threshold.

        Returns:
            jnp.ndarray: Probability in [0, 1].
        """
        alive = jnp.array(1.0)
        for excess, temperature in violations:
            alive = alive * (1.0 - jax.nn.sigmoid(excess / temperature))
        return 1.0 - alive

    @staticmethod
    def hard_any(violations, backend: ModuleType):
        """
        ``any(excess > 0)`` over (excess, temperature) pairs.
        """
        is_absorbing = backend.array(False)
        for excess, _ in violations:
            is_absorbing = backend.logical_or(is_absorbing, backend.greater(excess, 0.0))
        return is_absorbing

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType,
                             traj=None) -> Union[bool, Any]:
        """
        Check if the current state is terminal. Compatible with both CPU Mujoco and Mjx.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): Observations with shape (n_samples, n_obs).
            info (Dict[str, Any]): The info dictionary.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): Additional carry information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def register(cls) -> None:
        """
        Register a TerminalStateHandler in the TerminalStateHandler list.

        Adds the handler to the `registered` dictionary if not already present.
        """
        env_name = cls.__name__

        if env_name not in TerminalStateHandler.registered:
            TerminalStateHandler.registered[env_name] = cls
