from typing import Any, Dict, List, Union, Tuple
from types import ModuleType
from copy import deepcopy

import numpy as np
import jax
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model


from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.stateful_object import StatefulObject


class ControlFunction(StatefulObject):
    """
    Base class for all control functions.
    """

    registered: Dict[str, type] = dict()

    def __init__(self, env: any, low: np.ndarray, high: np.ndarray, **kwargs: Dict):
        """
        Initialize the control function class.

        Args:
            env (Any): The environment instance.
            low (np.ndarray): The lower bound of the action space.
            high (np.ndarray): The upper bound of the action space.
            **kwargs (Dict): Additional keyword arguments.
        """
        self._low = low
        self._high = high

        # compute the controller frequency
        n_intermediate_steps = env._n_intermediate_steps
        env_frequency = 1 / env.simulation_dt
        self._controller_frequency = env_frequency * n_intermediate_steps

    def reset(self, env: Any,
            model: Union[MjModel, Model],
            data: Union[MjData, Data],
            carry: Any,
            backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the control function.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def generate_action(self, env: Any,
                        action: Union[np.ndarray, jax.Array],
                        model: Union[MjModel, Model],
                        data: Union[MjData, Data],
                        carry: Any,
                        backend: ModuleType) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """
        Call the action with control function.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jax.Array]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The generated action and carry.

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    @property
    def frequency(self):
        """
        Get the controller frequency. This can differ from the environment frequency if intermediate steps are used.
        """
        return self._controller_frequency

    @property
    def run_with_simulation_frequency(self):
        """
        If true, the control function is called with the simulation frequency.
        """
        return False

    @property
    def action_limits(self):
        """
        Get the action space limits defined by the controller.
        """
        return deepcopy(self._low), deepcopy(self._high)

    @staticmethod
    def _get_actuator_limits(action_indices, model):
        """
        Returns the actuator control ranges of the model.

         Args:
             action_indices (list): A list of actuator indices.
             model: MuJoCo model.

         Returns:
             Two nd.ndarrays defining the action space limits.

         """
        low = []
        high = []
        for index in action_indices:
            if model.actuator_ctrllimited[index]:
                low.append(model.actuator_ctrlrange[index][0])
                high.append(model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)

        return np.array(low), np.array(high)

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the control function class.

        Returns:
            str: The name of the control function class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a control function class.

        Raises:
            ValueError: If the control function is already registered.
        """
        cls_name = cls.get_name()

        if cls_name in ControlFunction.registered:
            raise ValueError(f"ControlFunction '{cls_name}' is already registered.")

        ControlFunction.registered[cls_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered control functions.

        Returns:
            List[str]: A list of registered control function class names.
        """
        return list(ControlFunction.registered.keys())
