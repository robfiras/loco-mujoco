"""Static terrains: the flat floor that is part of every model."""

from typing import Any, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class StaticTerrain(Terrain):
    """
    Static terrain class inheriting from Terrain. This class is used for terrains that do not change over time
    (e.g., flat terrain). Used as is, it keeps the model's own floor, a plane at z = 0.

    """

    def reset(self, env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.
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
        Update the terrain.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).
            traj (Trajectory, optional): The trajectory of the environment, if any.

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.
        """
        assert_backend_is_supported(backend)
        return model, data, carry

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The unmodified simulation specification.
        """
        return spec

    @property
    def is_dynamic(self) -> bool:
        """
        Check if the terrain is dynamic.

        Returns:
            bool: False, as this terrain is static.
        """
        return False

    def get_height_at(self, xy: Union[np.ndarray, jnp.ndarray], backend: ModuleType, env: Any = None,
                      carry: Any = None) -> Union[np.ndarray, jnp.ndarray]:
        """
        Height of the flat floor (z = 0) at world points.

        Args:
            xy (Union[np.ndarray, jnp.ndarray]): World xy positions, shape ``(..., 2)``.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            env (Any): The environment instance (unused).
            carry (Any): Additional carry (unused).

        Returns:
            Union[np.ndarray, jnp.ndarray]: Zeros of shape ``(...)``.

        """
        assert_backend_is_supported(backend)
        return backend.zeros(xy.shape[:-1])


class FlatTerrain(StaticTerrain):
    """
    The default terrain: the model's own floor, a plane at z = 0. It is a :class:`StaticTerrain` under the name of
    what it looks like. ``terrain_type="StaticTerrain"`` selects the same terrain.

    """
