from typing import Any, Union, List, Tuple
from types import ModuleType

import jax
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported
from loco_mujoco.core.initial_state_handler.base import InitialStateHandler


class TrajInitialStateHandler(InitialStateHandler):
    """
    Initial state handler setting the initial joint positions and velocities from a trajectory.
    At reset, the handler sets the initial state of the simulation to the state of the trajectory.
    To control the state of the trajectory, the trajectory handler params need to be set in the environment.

    """

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

        assert traj is not None, "If TrajInitialStateHandler is used, traj_data must be passed."
        assert env.th is not None, "If TrajInitialStateHandler is used, a trajectory has to be loaded."

        # Get the current trajectory data sample
        traj_data_sample = env.th.get_current_traj_data(traj.data, carry, backend)

        if backend == np:
            data = env.set_sim_state_from_traj_data(data, traj_data_sample, carry)
        else:
            data = env.mjx_set_sim_state_from_traj_data(data, traj_data_sample, carry)

        return data, carry
