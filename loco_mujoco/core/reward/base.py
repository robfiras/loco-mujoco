from types import ModuleType
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.trajectory.handler import TrajectoryHandler


class Reward(StatefulObject):
    """
    Interface to specify a reward function.
    """

    registered: Dict[str, type] = dict()

    def __init__(self,
                 env: Any,
                 **kwargs: Any):
        """
        Initialize the Reward class.

        Args:
            env (Any): Environment instance.
            **kwargs (Any): Additional keyword arguments.
        """
        self._obs_container = env.obs_container
        self._info_props = env._get_all_info_properties()
        self.initialized: bool = False

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the reward class.

        Returns:
            str: The name of the reward class.
        """
        return cls.__name__

    def init_from_traj(self, traj_handler: TrajectoryHandler = None) -> None:
        """
        Initialize the reward class from a trajectory.

        Args:
            traj_handler (TrajectoryHandler, optional): The trajectory handler. Defaults to None.
        """
        pass

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType,
                 traj_model=None,
                 traj_data=None) -> Tuple[float, Any]:
        """
        Compute the reward.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj_model=None,
              traj_data=None):
        """
        Reset the reward.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        """
        return data, carry

    @classmethod
    def register(cls) -> None:
        """
        Register a reward in the reward list.
        """
        env_name = cls.get_name()

        if env_name not in Reward.registered:
            Reward.registered[env_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered rewards.

        Returns:
            List[str]: The list of registered rewards.
        """
        return list(Reward.registered.keys())

    @property
    def requires_trajectory(self) -> bool:
        """
        Check if the reward requires trajectory data.

        Returns:
            bool: True if trajectory data is required, False otherwise.
        """
        return False
