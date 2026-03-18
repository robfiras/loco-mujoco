from typing import Any, Union, Dict, List, Tuple
from types import ModuleType

from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils import assert_backend_is_supported


class InitialStateHandler(StatefulObject):
    """ Base class for initial state handlers. """
    registered: Dict[str, type] = dict()

    def __init__(self, env: any, **kwargs: Dict):
        """
        Initialize the InitialStateHandler class.

        Args:
            env (Any): The environment instance.
            **kwargs (Dict): Additional keyword arguments.
        """
        pass

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
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            ValueError: If the backend module is not supported.
            NotImplementedError: If the method is not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the init state handler class.

        Returns:
            str: The name of the init state handler class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a init state handler class.

        Raises:
            ValueError: If the init state handler is already registered.
        """
        cls_name = cls.get_name()

        if cls_name in InitialStateHandler.registered:
            raise ValueError(f"InitStateHandler '{cls_name}' is already registered.")

        InitialStateHandler.registered[cls_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered init state handler.

        Returns:
            List[str]: A list of registered init state handler class names.
        """
        return list(InitialStateHandler.registered.keys())
