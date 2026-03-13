from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.trajectory import TrajectoryHandler
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
              traj_model=None,
              traj_data=None) -> Tuple[Union[MjData, Data], Any]:
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
                     traj_model=None,
                     traj_data=None) -> Union[bool, Any]:
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
                         traj_model=None,
                         traj_data=None) -> Union[bool, Any]:
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

    def _is_absorbing_compat(self,
                             env: Any,
                             obs: Union[np.ndarray, jnp.ndarray],
                             info: Dict[str, Any],
                             data: Union[MjData, Data],
                             carry: Any,
                             backend: ModuleType,
                             traj_model=None,
                             traj_data=None) -> Union[bool, Any]:
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

    def init_from_traj(self, th: TrajectoryHandler) -> None:
        """
        Initialize the TerminalStateHandler from a trajectory handler (optional).

        Args:
            th (TrajectoryHandler): The trajectory handler containing the trajectory.
        """
        pass

    @classmethod
    def register(cls) -> None:
        """
        Register a TerminalStateHandler in the TerminalStateHandler list.

        Adds the handler to the `registered` dictionary if not already present.
        """
        env_name = cls.__name__

        if env_name not in TerminalStateHandler.registered:
            TerminalStateHandler.registered[env_name] = cls
