from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.observations.base import StatefulObservation
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid


class HeightScan(StatefulObservation):
    """
    Terrain heights on a grid around the root, in the root's heading frame (Isaac Lab / mjlab style height scan).

    Each entry is ``root_height - terrain_height(point) - offset``, clipped to ``clip``. The grid spans ``size``
    (forward, lateral) metres with ``resolution`` spacing, centred ``forward_offset`` metres ahead of the root.
    The heights come from the terrain's ``get_height_at``, so the observation works with all built-in terrains,
    also in jitted code on MJX and MjWarp.

    Args:
        obs_name: The name of the observation.
        size: Grid extent (forward, lateral) in metres.
        resolution: Grid spacing in metres.
        forward_offset: Shift of the grid centre along the heading, in metres.
        offset: Subtracted from every entry (e.g. the nominal root height), in metres.
        clip: (low, high) clipping range of the output.
        xml_name: Name of the free joint giving the root pose. Defaults to the env's root free joint.
        differentiable: Let gradients flow through the scan into the root pose. Off by default: for methods that
            backpropagate through the observation (e.g. SuperMimic's BPTT through a world model), step edges turn the
            scan into near-discontinuities whose gradients with respect to the root position are spurious.
    """

    def __init__(self, obs_name: str, size: Sequence[float] = (1.6, 1.0), resolution: float = 0.1,
                 forward_offset: float = 0.0, offset: float = 0.0, clip: Sequence[float] = (-1.0, 1.0),
                 xml_name: str = None, differentiable: bool = False, **kwargs):
        super().__init__(obs_name, **kwargs)
        self._differentiable = differentiable
        nx = int(round(size[0] / resolution)) + 1
        ny = int(round(size[1] / resolution)) + 1
        gx = np.linspace(-size[0] / 2, size[0] / 2, nx) + forward_offset
        gy = np.linspace(-size[1] / 2, size[1] / 2, ny)
        self._grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)   # (n, 2) in heading frame
        self.dim = len(self._grid)
        self._offset = offset
        self._clip = tuple(clip)
        self._xml_name = xml_name

    def _init_from_mj(self, env, model, data, current_obs_size):
        # the terrain is set up after the observations, so it is only looked up when the observation is computed
        name = env.root_free_joint_xml_name if self._xml_name is None else self._xml_name
        self._qpos_ids = np.array(mj_jntname2qposid(name, model))
        self.min, self.max = [self._clip[0]] * self.dim, [self._clip[1]] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        q = data.qpos[self._qpos_ids]
        w, x, y, z = q[3], q[4], q[5], q[6]
        yaw = backend.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = backend.cos(yaw), backend.sin(yaw)
        g = backend.asarray(self._grid)
        pts = backend.stack([q[0] + c * g[:, 0] - s * g[:, 1], q[1] + s * g[:, 0] + c * g[:, 1]], axis=-1)
        h = env._terrain.get_height_at(pts, backend, env=env, carry=carry)
        scan = backend.clip(q[2] - h - self._offset, *self._clip)
        if backend == jnp and not self._differentiable:
            scan = jax.lax.stop_gradient(scan)
        return scan, carry
