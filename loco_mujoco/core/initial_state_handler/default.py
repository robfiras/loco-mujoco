from typing import Any, Union, List, Tuple
from types import ModuleType

import jax
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported
from loco_mujoco.core.initial_state_handler.base import InitialStateHandler


class DefaultInitialStateHandler(InitialStateHandler):

    """
    Basic initial state handler setting the initial joint positions and velocities. By default,
    the initial joint positions and velocities are set to None, which means that the initial state
    is not modified.

    """

    def __init__(self, env: Any, qpos_init=None, qvel_init=None):
        """
        Initialize the DefaultInitialStateHandler class.

        Args:
            env (Any): The environment instance.
            qpos_init (Union[None, List[float]]): Initial joint positions.
            qvel_init (Union[None, List[float]]): Initial joint velocities.
        """

        self.qpos_init = np.array(qpos_init) if qpos_init is not None else None
        self.qvel_init = np.array(qvel_init) if qvel_init is not None else None

        super().__init__(env)

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the init state handler with its state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
        """
        assert_backend_is_supported(backend)

        if self.qpos_init is not None:
            data = self.set_qpos(self.qpos_init, data, backend)
        if self.qvel_init is not None:
            data = self.set_qvel(self.qvel_init, data, backend)

        return data, carry

    @staticmethod
    def set_qpos(qpos: Union[np.ndarray, jax.Array],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """
        Set the joint positions in the simulation data.

        Args:
            qpos (List[float]): The joint positions.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        if backend == np:
            data.qpos[:] = qpos
        else:
            data = data.replace(qpos=data.qpos.at[:].set(qpos))

        return data

    @staticmethod
    def set_qvel(qvel: Union[np.ndarray, jax.Array],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """
        Set the joint velocities in the simulation data.

        Args:
            qvel (List[float]): The joint velocities.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        if backend == np:
            data.qvel[:] = qvel
        else:
            data = data.replace(qvel=data.qvel.at[:].set(qvel))

        return data
