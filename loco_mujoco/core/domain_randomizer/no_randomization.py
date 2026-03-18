from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class NoDomainRandomization(DomainRandomizer):
    """
    A domain randomizer that performs no randomization.
    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset with no randomization applied.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The unchanged data and carry.
        """
        assert_backend_is_supported(backend)
        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType,
               traj=None) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update with no randomization applied.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The unchanged model, data, and carry
        """
        assert_backend_is_supported(backend)
        return model, data, carry

    def update_observation(self, env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType,
                           traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the observation with no randomization applied.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The unchanged observation and carry.
        """
        assert_backend_is_supported(backend)
        return obs, carry

    def update_action(self, env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType,
                      traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with no randomization applied.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The unchanged action and carry.
        """
        assert_backend_is_supported(backend)
        return action, carry
