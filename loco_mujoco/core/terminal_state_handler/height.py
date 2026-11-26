"""Terminal state handler based on the height of the root."""

from typing import Dict, Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model


from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class HeightBasedTerminalStateHandler(TerminalStateHandler):
    """
    Check if the current state is terminal based on the height of the root.
    """

    soft_temperature_m = 0.02  # width of the smooth termination around the height limits, in meters
    def __init__(self, env: Any, **handler_config: Dict[str, Any]):
        """
        Initialize the TerminalStateHandler.

        Args:
            env (Any): The environment instance.
            **handler_config (Any): Configuration dictionary.
        """
        super().__init__(env, **handler_config)

        self.root_height_range = self._info_props["root_height_healthy_range"]
        self.root_free_joint_xml_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"],
                                                                  env._model))

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
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        return data, carry

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
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=np)

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
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp)

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
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Union[bool, Any]: Whether the current state is terminal, and the carry.

        """
        return self.hard_any(self._violations(data), backend), carry

    def mjx_absorbing_prob(self, env: Any, obs: jnp.ndarray, info: Dict[str, Any], data: Data, carry: Any,
                           traj=None) -> jnp.ndarray:
        """
        Smooth termination probability of the root height range, see ``TerminalStateHandler.soft_any``.

        Args:
            env (Any): The environment instance.
            obs (jnp.ndarray): Observation of the environment.
            info (Dict[str, Any]): The info dictionary.
            data (Data): The Mujoco data structure for Mjx.
            carry (Any): Additional carry information.
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            jnp.ndarray: Probability in [0, 1] that the current state is terminal.

        """
        return self.soft_any(self._violations(data))

    def _violations(self, data: Union[MjData, Data]):
        """
        (excess, temperature) pairs of the conditions: the root height below and above the healthy range.
        """
        height = data.qpos[self.root_free_joint_xml_ind][2]
        return [(self.root_height_range[0] - height, self.soft_temperature_m),
                (height - self.root_height_range[1], self.soft_temperature_m)]
