from typing import Any, Union, List, Tuple
from types import ModuleType
import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjModel, MjData
from mujoco.mjx import Model, Data

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast


def extract_z_rotation(global_rotation, backend):
    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # Convert to Euler angles
    euler_angles = global_rotation.as_euler('xyz', degrees=False)

    # Isolate yaw (z-axis rotation). Use `backend.array` so we pass an ndarray
    # (not a Python list) to R.from_euler — newer JAX rejects lists.
    yaw_only = backend.array([0.0, 0.0, euler_angles[2]])

    # Convert back to rotation object
    z_rotation = R.from_euler('xyz', yaw_only, degrees=False)

    return z_rotation


class RootVelocityArrowVisualizer:
    """
    A class to visualize the root velocity and rotational velocity arrows in
    a carry state to be visualized in simulation.

    Attributes:
        _arrow_n_visual_geoms (int): Number of visual geometries for arrows.
        _z_offset (np.ndarray): Z-axis offset for visualization.
        _arrow_color (np.ndarray): Color of the linear velocity arrow.
        _arrow_type (int): Geometry type for the arrow.
        _rot_vel_arrow_type (int): Geometry type for the rotational velocity arrow.
        _rot_vel_arrow_size (np.ndarray): Size of the rotational velocity arrow.
        _rot_vel_arrow_color (np.ndarray): Color of the rotational velocity arrow.
        _rot_vel_point_size (np.ndarray): Size of the rotational velocity point.
        _center_sphere_size (np.ndarray): Size of the center sphere.
        _center_sphere_type (int): Geometry type for the center sphere.
        _center_sphere_mat (np.ndarray): Material of the center sphere.
        _center_sphere_color (np.ndarray): Color of the center sphere.
    """

    def __init__(self, info_props, visualize_rot_vel: bool = False):
        self._arrow_n_visual_geoms = 3 if visualize_rot_vel else 2
        self._visualize_rot_vel = visualize_rot_vel
        self._z_offset = np.array([0.0, 0.0, 0.3]) + info_props["goal_visualization_arrow_offset"]
        self._arrow_color = np.array([1.0, 0.0, 0.0, 0.75])
        self._arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
        self._rot_vel_arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW1)
        self._rot_vel_arrow_size = np.array([0.015, 0.015, 0.5])
        self._rot_vel_arrow_color = np.array([1.0, 1.0, 0.0, 0.75])
        self._rot_vel_point_size = np.array([0.025, 0.025, 0.025])
        self._center_sphere_size = np.array([0.05, 0.0, 0.0])
        self._center_sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        self._center_sphere_mat = np.eye(3).reshape(-1)
        self._center_sphere_color = np.array([0.063, 0.0, 0.541, 1.0])

    def set_visuals(self,
                    goal: Union[np.ndarray, jnp.ndarray],
                    env: Any,
                    model: Union[MjModel, Model],
                    data: Union[MjData, Data],
                    carry: Any,
                    root_body_id: int,
                    free_jnt_qposid: Union[np.ndarray, jnp.ndarray],
                    visual_geoms_idx: List[int],
                    backend: ModuleType) -> Any:
        """
        Sets the visuals for root velocity and rotational velocity in the carry.

        Args:
            goal (Union[np.ndarray, jnp.ndarray]): Target velocity in global frame.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[Model, Data]): The simulation data.
            carry (Any): Additional state information for rendering.
            root_body_id (int): ID of the root body in the simulation.
            free_jnt_qposid (Union[np.ndarray, jnp.ndarray],): Index for the root joint position in `qpos`.
            visual_geoms_idx (List[int]): Indices of visual geometries.
            backend (ModuleType): Backend module (e.g., `np` or `jnp`) for calculations.

        Returns:
            Any: Updated carry with the new visualizations set.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        user_scene = carry.user_scene
        geoms = user_scene.geoms

        # Get root orientation
        root_qpos = backend.squeeze(data.qpos[free_jnt_qposid])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))
        root_quat = extract_z_rotation(root_quat, backend)
        root_mat = root_quat.as_matrix()

        # Get root position
        root_pos = data.xpos[root_body_id]

        # Goal velocity in local frame
        goal_lin_vel = backend.concatenate([goal[:3]])
        goal_vel_local = root_mat @ goal_lin_vel
        goal_rot_vel = goal[5]

        # Base rotation arrow (add 90-degree rotation around y-axis)
        arrow_mat = root_mat @ R.from_euler("y", 90, degrees=True).as_matrix()

        # Calculate angle between two arrays
        v1 = backend.array([0.0, 0.0, 1.0])
        reorder = np.array([2, 1, 0])
        goal_lin_vel_norm = goal_lin_vel[reorder] / backend.linalg.norm(goal_lin_vel[reorder])
        cross = backend.cross(v1, goal_lin_vel_norm)
        dot = backend.dot(v1, goal_lin_vel_norm)
        angle = backend.arctan2(cross[0], dot)  # Use x-component for sign
        arrow_mat = arrow_mat @ R.from_euler("x", angle, degrees=False).as_matrix()
        arrow_mat = arrow_mat.reshape(-1)

        min_arrow_length = 0.1
        arrow_length = backend.linalg.norm(goal_lin_vel) + min_arrow_length

        # Calculate position of the point indicating rotational velocity
        scaling = 0.15
        local_x_y_rot_vel = backend.array([
            backend.cos(goal_rot_vel * (backend.pi / 2)) * scaling,
            backend.sin(goal_rot_vel * (backend.pi / 2)) * scaling,
            self._z_offset[2]
        ])

        # Rotational velocity arrow matrix
        if self._visualize_rot_vel:
            rot_vel_arrow_mat = root_mat @ R.from_euler("y", 90, degrees=True).as_matrix()
            local_x_y_rot_vel_norm = local_x_y_rot_vel[reorder] / backend.linalg.norm(local_x_y_rot_vel[reorder])
            cross = backend.cross(v1, local_x_y_rot_vel_norm)
            dot = backend.dot(v1, local_x_y_rot_vel_norm)
            angle = backend.arctan2(cross[0], dot)  # Use x-component for sign
            rot_vel_arrow_mat = rot_vel_arrow_mat @ R.from_euler("x", angle, degrees=False).as_matrix()
            rot_vel_arrow_mat = rot_vel_arrow_mat.reshape(-1)

        # Arrow size
        arrow_size = backend.array([0.025, 0.025, arrow_length])

        if backend == jnp:
            # Set visualizations using JAX's array operations
            arrow_idx = visual_geoms_idx[0]
            geom_pos = user_scene.geoms.pos.at[arrow_idx].set(root_pos + self._z_offset)
            geom_mat = user_scene.geoms.mat.at[arrow_idx].set(arrow_mat)
            geom_type = user_scene.geoms.type.at[arrow_idx].set(self._arrow_type)
            geom_size = user_scene.geoms.size.at[arrow_idx].set(arrow_size)
            geom_rba = user_scene.geoms.rgba.at[arrow_idx].set(self._arrow_color)

            center_sphere_idx = visual_geoms_idx[1]
            geom_pos = geom_pos.at[center_sphere_idx].set(root_pos + self._z_offset)
            geom_mat = geom_mat.at[center_sphere_idx].set(self._center_sphere_mat)
            geom_type = geom_type.at[center_sphere_idx].set(self._center_sphere_type)
            geom_size = geom_size.at[center_sphere_idx].set(self._center_sphere_size)
            geom_rba = geom_rba.at[center_sphere_idx].set(self._center_sphere_color)

            if self._visualize_rot_vel:
                rot_vel_arrow_idx = visual_geoms_idx[2]
                geom_pos = geom_pos.at[rot_vel_arrow_idx].set(root_pos + self._z_offset)
                geom_mat = geom_mat.at[rot_vel_arrow_idx].set(rot_vel_arrow_mat)
                geom_type = geom_type.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_type)
                geom_size = geom_size.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_size)
                geom_rba = geom_rba.at[rot_vel_arrow_idx].set(self._rot_vel_arrow_color)

        else:
            # Set visualizations using NumPy's array operations
            arrow_idx = visual_geoms_idx[0]
            user_scene.geoms.pos[arrow_idx] = root_pos + self._z_offset
            user_scene.geoms.mat[arrow_idx] = arrow_mat
            user_scene.geoms.type[arrow_idx] = self._arrow_type
            user_scene.geoms.size[arrow_idx] = arrow_size
            user_scene.geoms.rgba[arrow_idx] = self._arrow_color

            center_sphere_idx = visual_geoms_idx[1]
            user_scene.geoms.pos[center_sphere_idx] = root_pos + self._z_offset
            user_scene.geoms.mat[center_sphere_idx] = self._center_sphere_mat
            user_scene.geoms.type[center_sphere_idx] = self._center_sphere_type
            user_scene.geoms.size[center_sphere_idx] = self._center_sphere_size
            user_scene.geoms.rgba[center_sphere_idx] = self._center_sphere_color

            if self._visualize_rot_vel:
                rot_vel_arrow_idx = visual_geoms_idx[2]
                user_scene.geoms.pos[rot_vel_arrow_idx] = root_pos + self._z_offset
                user_scene.geoms.mat[rot_vel_arrow_idx] = rot_vel_arrow_mat
                user_scene.geoms.type[rot_vel_arrow_idx] = self._rot_vel_arrow_type
                user_scene.geoms.size[rot_vel_arrow_idx] = self._rot_vel_arrow_size
                user_scene.geoms.rgba[rot_vel_arrow_idx] = self._rot_vel_arrow_color

            geom_pos = user_scene.geoms.pos[self.visual_geoms_idx]
            geom_mat = user_scene.geoms.mat[self.visual_geoms_idx]
            geom_type = user_scene.geoms.type[self.visual_geoms_idx]
            geom_size = user_scene.geoms.size[self.visual_geoms_idx]
            geom_rba = user_scene.geoms.rgba[self.visual_geoms_idx]

        # Update carry
        new_user_scene = user_scene.replace(geoms=user_scene.geoms.replace(
            pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rba))
        carry = carry.replace(user_scene=new_user_scene)

        return carry
