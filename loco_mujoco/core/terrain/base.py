from typing import Dict, Any, List, Union, Tuple
from types import ModuleType

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.backend import assert_backend_is_supported


class Terrain(StatefulObject):
    """
    Base interface for all terrain classes.

    The environment drives a terrain through these hooks, identically on MuJoCo (CPU, NumPy backend) and on MJX /
    MuJoCo Warp (JAX backend, one env per vmapped world):

        - ``modify_spec``: once, while the model is built. Adds the terrain's assets and geoms.
        - ``init_state``: per env, when the carry is created. Returns the terrain's per-env state
          (``carry.terrain_state``).
        - ``reset``: per env, on every reset, after the trajectory state has been reset and before the initial state
          is set. May change the data and the carry, not the model.
        - ``shift_to_env_origin`` (if ``has_env_origins``) and ``ground_robot`` (if ``grounds_robot``): per env, on
          every reset, after the initial state has been set.
        - ``update``: per env, before every physics step. The model is not stored per env: the env passes the same
          base model every step, so a terrain that changes the model per env (e.g. a heightfield or geoms of its
          own) re-applies the change here from its state in the carry, as the domain randomizer does.
        - ``get_height_at``: terrain height at world points, for observations (``HeightScan``) and reset logic.

    Placement of the robot: by default every episode starts at the world origin in xy (the environment recenters
    the trajectory, ``recenter_root_xy``). A terrain that spreads envs over the map returns ``has_env_origins``
    and moves each env to its own origin in ``shift_to_env_origin``. A terrain whose map is built around the clips
    of a dataset returns ``trajectories_in_world_frame``, and episodes then start where the clip is.
    """

    registered: Dict[str, type] = dict()

    # If the hfield is generated dynamically, this should be set to True to update the hfield in the viewer.
    viewer_needs_to_update_hfield: bool = False

    def __init__(self, env: Any, **terrain_config: [str, Any]):
        """
        Initialize the Terrain class.

        Args:
            env (Any): The environment instance.
            **terrain_config (Any): Configuration parameters for the terrain.
        """
        self.terrain_conf = terrain_config

        # The heightfield the viewer re-uploads for dynamic terrains (see ``is_dynamic``). Set in ``modify_spec``.
        self.hfield_id: Union[int, None] = None

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
            traj: The trajectory in use, if any.

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType,
               traj=None) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the terrain before a physics step. Per-env changes of the model have to be applied here, every step
        (see the class docstring).

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).
            traj: The trajectory in use, if any.

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The modified simulation specification.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def get_height_at(self, xy: Union[np.ndarray, jnp.ndarray], backend: ModuleType, env: Any = None,
                      carry: Any = None) -> Union[np.ndarray, jnp.ndarray]:
        """
        Height of the terrain's collision surface at world points.

        Args:
            xy (Union[np.ndarray, jnp.ndarray]): World xy points, shape (..., 2).
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).
            env (Any): The environment instance. Terrains may read their data from its model.
            carry (Any): Carry instance of the env. Needed by terrains that differ per env.

        Returns:
            Union[np.ndarray, jnp.ndarray]: The heights, shape (...).

        Raises:
            NotImplementedError: If the terrain does not provide heights.
        """
        raise NotImplementedError(f"{self.get_name()} does not provide terrain heights.")

    @property
    def has_env_origins(self) -> bool:
        """
        Whether envs are placed at per-env origins on the terrain (e.g., a shared tiled terrain).

        Returns:
            bool: False by default; all envs start around the world origin.
        """
        return False

    def get_env_origin_xy(self, carry: Any, backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Get the world xy origin of the env, i.e., where the trajectory start (or default pose) is placed.

        Args:
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[np.ndarray, jnp.ndarray]: The xy origin, shape (2,).
        """
        return backend.zeros(2)

    def shift_to_env_origin(self, env: Any, data: Union[MjData, Data], carry: Any,
                            backend: ModuleType) -> Union[MjData, Data]:
        """
        Move the env's free joints to its origin. Called after the initial state has been set on reset.

        Args:
            env (Any): The environment instance.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The (unchanged by default) simulation data.
        """
        return data

    @property
    def trajectories_in_world_frame(self) -> bool:
        """
        Whether the terrain is built around the clips of a dataset, each clip already placed on its part of the map.
        Episodes then start where their clip is instead of at the world origin (the environment does not recenter
        the trajectory, see ``recenter_root_xy`` of ``LocoEnv``).

        Returns:
            bool: False by default.
        """
        return False

    @property
    def grounds_robot(self) -> bool:
        """
        Whether the terrain lifts the robot out of the ground after the initial state has been set on reset (see
        ``ground_robot``). False by default: a flat floor needs no correction.
        """
        return False

    def ground_robot(self, env: Any, data: Union[MjData, Data], backend: ModuleType) -> Union[MjData, Data]:
        """
        Lift the robot so that none of its collision geometry starts inside the terrain. Called on reset after the
        initial state (including reset noise) has been set.

        Args:
            env (Any): The environment instance.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The (unchanged by default) simulation data.
        """
        return data

    @property
    def is_dynamic(self) -> bool:
        """
        Whether the terrain writes heightfield data per env at runtime (``hfield_data``). The viewer then re-uploads
        the heightfield ``hfield_id``, and MuJoCo Warp, which keeps one ``hfield_data`` for all worlds, refuses the
        terrain. Terrains that change other model fields per env (e.g. geoms) are not dynamic in this sense.

        Returns:
            bool: True if the terrain is dynamic, False otherwise.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    @property
    def requires_spec_modification(self) -> bool:
        """
        Check if the terrain requires modification of the simulation specification.

        Returns:
            bool: True if the terrain requires specification modification, False otherwise.
        """
        return self.__class__.modify_spec != Terrain.modify_spec

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the terrain class.

        Returns:
            str: The name of the terrain class.
        """
        return cls.__name__

    @classmethod
    def register(cls):
        """
        Register a terrain class.

        Raises:
            ValueError: If a terrain with the same name is already registered.
        """
        cls_name = cls.get_name()

        if cls_name in Terrain.registered:
            raise ValueError(f"Terrain '{cls_name}' is already registered.")

        Terrain.registered[cls_name] = cls

    @staticmethod
    def list_registered() -> List[str]:
        """
        List registered terrain classes.

        Returns:
            List[str]: A list of registered terrain class names.
        """
        return list(Terrain.registered.keys())

    @staticmethod
    def _set_attribute_in_model(model: Union[MjModel, Model],
                                attribute: str,
                                value: Any,
                                backend: ModuleType) -> Union[MjModel, Model]:
        """
        Set an attribute in the model. This works for both NumPy and JAX backends.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            attribute (str): The attribute to set.
            value (Any): The value to assign to the attribute.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjModel, Model]: The updated model.
        """
        assert_backend_is_supported(backend)

        if backend == jnp:
            model = model.tree_replace({attribute: value})
        else:
            setattr(model, attribute, value)
        return model


def hfield_surface_height(H: Union[np.ndarray, jnp.ndarray], fx: Union[np.ndarray, jnp.ndarray],
                          fy: Union[np.ndarray, jnp.ndarray], nrow: int, ncol: int, backend: ModuleType,
                          adr: Union[int, np.ndarray, jnp.ndarray] = 0,
                          nearest: bool = False) -> Union[np.ndarray, jnp.ndarray]:
    """
    Surface of a MuJoCo heightfield at fractional sample coordinates.

    MuJoCo collides with (and ray-casts against) two triangles per heightfield cell, split along the diagonal from
    sample (r, c) to (r + 1, c + 1); this interpolates on those triangles, so it is exact, unlike bilinear
    interpolation.

    Args:
        H: Flat row-major samples; row r runs along y, column c along x.
        fx, fy: Fractional column and row coordinates of the points, shape (...). Clamped to the heightfield.
        nrow, ncol: Rows and columns of the heightfield (both at least 2).
        backend: numpy or jax.numpy.
        adr: Index of sample (0, 0) in H (a scalar, or one per point).
        nearest: Return the nearest sample instead.

    Returns:
        The interpolated sample values, shape (...).
    """
    H = H.reshape(-1)
    fx = backend.clip(fx, 0.0, ncol - 1.0)
    fy = backend.clip(fy, 0.0, nrow - 1.0)
    c0 = backend.clip(backend.floor(fx).astype(int), 0, ncol - 2)
    r0 = backend.clip(backend.floor(fy).astype(int), 0, nrow - 2)
    tx, ty = fx - c0, fy - r0
    if nearest:
        tx, ty = backend.round(tx), backend.round(ty)
    i = adr + r0 * ncol + c0
    h00, h01, h10, h11 = H[i], H[i + 1], H[i + ncol], H[i + ncol + 1]
    return backend.where(tx >= ty, h00 + tx * (h01 - h00) + ty * (h11 - h01),
                         h00 + ty * (h10 - h00) + tx * (h11 - h10))
