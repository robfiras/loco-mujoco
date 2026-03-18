from types import ModuleType
from typing import Any, Union, Dict, Tuple
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model
import scipy as np_scipy
import jax.scipy as jnp_scipy

from loco_mujoco.core.terrain import DynamicTerrain
from loco_mujoco.core.utils import mj_jntname2qposid
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class RoughTerrainState:
    """
    Represents the state of the rough terrain.

    Attributes:
        height_field_raw (Union[np.ndarray, jax.Array]): The raw height field data.
    """
    height_field_raw: Union[np.ndarray, jax.Array]


class RoughTerrain(DynamicTerrain):
    """
    Dynamic rough terrain class for simulating uneven surfaces. This terrain is generated randomly on
    each reset.

    """

    viewer_needs_to_update_hfield: bool = True

    def __init__(self, env: Any,
                 inner_platform_size_in_meters: float = 1.0,
                 random_min_height: float = -0.05,
                 random_max_height: float = 0.05,
                 random_step: float = 0.005,
                 random_downsampled_scale: float = 0.4,
                 **kwargs: Any):
        """
        Initialize the rough terrain.

        Args:
            env (Any): The environment instance.
            inner_platform_size_in_meters (float): Size of the inner platform in meters.
            random_min_height (float): Minimum random height for terrain generation.
            random_max_height (float): Maximum random height for terrain generation.
            random_step (float): Step size for random height values.
            random_downsampled_scale (float): Downsample scale for terrain.
            **kwargs (Any): Additional arguments for initialization.
        """
        super().__init__(env, **kwargs)

        self.inner_platform_size_in_meters = inner_platform_size_in_meters
        self.random_min_height = random_min_height
        self.random_max_height = random_max_height
        self.random_step = random_step
        self.random_downsampled_scale = random_downsampled_scale

        self.hfield_size = (4, 4, 30.0, 0.125)
        self.hfield_length = 80
        self.hfield_half_length_in_meters = self.hfield_size[0]
        self.max_possible_height = self.hfield_size[2]

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        self.heights_range = jnp.arange(self.random_min_height, self.random_max_height + self.random_step,
                                        self.random_step)

        self.x = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        self.y = jnp.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        x_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        y_upsampled = jnp.linspace(0, self.hfield_length, self.hfield_length)
        x_upsampled_grid, y_upsampled_grid = jnp.meshgrid(x_upsampled, y_upsampled, indexing='ij')
        self.points = jnp.stack([x_upsampled_grid.ravel(), y_upsampled_grid.ravel()], axis=1)
        platform_size = int(self.inner_platform_size_in_meters * self.one_meter_length)
        self.x1 = self.hfield_half_length - (platform_size // 2)
        self.y1 = self.hfield_half_length - (platform_size // 2)
        self.x2 = self.hfield_half_length + (platform_size // 2)
        self.y2 = self.hfield_half_length + (platform_size // 2)

        root_free_joint_xml_name = env.root_free_joint_xml_name
        self._free_jnt_qpos_id = np.array(mj_jntname2qposid(root_free_joint_xml_name, env._model))

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType,
                   traj=None) -> RoughTerrainState:
        """
        Initialize the state of the rough terrain.

        Args:
            env (Any): The environment instance.
            key (Any): JAX random key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            RoughTerrainState: The initialized terrain state.
        """
        assert_backend_is_supported(backend)
        return RoughTerrainState(backend.zeros((self.hfield_length, self.hfield_length)))

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """
        Modify the simulation specification to include the rough terrain.

        Args:
            spec (MjSpec): The simulation specification.

        Returns:
            MjSpec: The modified simulation specification.
        """
        from loco_mujoco.core.mujoco_base import Mujoco

        file_name = Mujoco.get_model_path("common", "default_hfield_80.png")
        spec.add_hfield(name='rough_terrain', size=self.hfield_size, file=file_name)
        for i, field in enumerate(spec.hfields):
            if field.name == 'rough_terrain':
                self.hfield_id = i
                break

        for g in spec.geoms:
            if g.name == 'floor':
                spec.delete(g)
                break

        wb = spec.worldbody
        wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='rough_terrain', group=2,
                    pos=(0, 0, -0.06), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1))

        return spec

    def reset(self, env: Any,
              model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the rough terrain and update its state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.
        """
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._jnp_random_uniform_terrain(_k), backend)
            carry = carry.replace(key=key)
        else:
            height_field_raw = self.isaac_hf_to_mujoco_hf(self._np_random_uniform_terrain(), backend)

        terrain_state = terrain_state.replace(height_field_raw=height_field_raw)
        carry = carry.replace(terrain_state=terrain_state)

        return data, carry

    def update(self, env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType,
               traj=None) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the rough terrain and simulation state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.
        """
        assert_backend_is_supported(backend)
        terrain_state = carry.terrain_state
        model = self._set_attribute_in_model(model, "hfield_data", terrain_state.height_field_raw, backend)
        data = self._reset_on_edge(data, backend)
        return model, data, carry

    def get_height_matrix(self, matrix_config: Dict[str, Any],
                          env: Any,
                          model: Union[MjModel, Model],
                          data: Union[MjData, Data],
                          carry: Any,
                          backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Get the height matrix of the terrain.

        Args:
            matrix_config (Dict[str, Any]): The matrix configuration.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[np.ndarray, jnp.ndarray]: The height matrix of the terrain.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        assert_backend_is_supported(backend)
        raise NotImplementedError

    def isaac_hf_to_mujoco_hf(self,
                              isaac_hf: Union[np.ndarray, jnp.ndarray],
                              backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        """
        Convert Isaac height field data to MuJoCo-compatible height field data.

        Args:
            isaac_hf (Union[np.ndarray, jnp.ndarray]): The Isaac height field data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[np.ndarray, jnp.ndarray]: The converted height field data.
        """
        assert_backend_is_supported(backend)

        hf = isaac_hf + backend.abs(backend.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def _np_random_uniform_terrain(self) -> np.ndarray:
        """
        Generate random uniform terrain using NumPy.

        Returns:
            np.ndarray: The generated height field.
        """
        add_height_field_downsampled = (
            np.random.choice(self.heights_range, size=(int(self.hfield_length * self.random_downsampled_scale),
                                                       int(self.hfield_length * self.random_downsampled_scale))))
        interpolator = np_scipy.interpolate.RegularGridInterpolator((self.x, self.y),
                                                                    add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field[self.x1:self.x2, self.y1:self.y2] = 0
        return add_height_field

    def _jnp_random_uniform_terrain(self, key: Any) -> jnp.ndarray:
        """
        Generate random uniform terrain using JAX.

        Args:
            key (Any): JAX random key.

        Returns:
            jnp.ndarray: The generated height field.
        """
        add_height_field_downsampled = (
            jax.random.choice(key, self.heights_range, shape=(int(self.hfield_length * self.random_downsampled_scale),
                                                              int(self.hfield_length * self.random_downsampled_scale))))
        interpolator = jnp_scipy.interpolate.RegularGridInterpolator((self.x, self.y),
                                                                     add_height_field_downsampled, method='linear')
        add_height_field = interpolator(self.points).reshape((self.hfield_length, self.hfield_length))
        add_height_field = add_height_field.at[self.x1:self.x2, self.y1:self.y2].set(0)
        return add_height_field

    def _reset_on_edge(self, data: Union[MjData, Data],
                       backend: ModuleType) -> Union[MjData, Data]:
        """
        Reset the robot position if it is on the edge of the terrain.

        Args:
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (e.g., numpy or jax.numpy).

        Returns:
            Union[MjData, Data]: The updated simulation data.
        """
        assert_backend_is_supported(backend)

        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        com_pos = data.qpos[self._free_jnt_qpos_id][:2]
        reached_edge = backend.array(((min_edge < backend.abs(com_pos[0])) & (backend.abs(com_pos[0]) < max_edge)) | (
                    (min_edge < backend.abs(com_pos[1])) & (backend.abs(com_pos[1]) < max_edge)))
        free_jnt_xy = self._free_jnt_qpos_id[:2]
        if backend == jnp:
            init_data = data.replace(qpos=data.qpos.at[free_jnt_xy].set(0.0))
            data = jax.lax.cond(reached_edge, lambda _: init_data, lambda _: data, None)
        else:
            if reached_edge:
                data.qpos[free_jnt_xy] = 0.0

        return data
