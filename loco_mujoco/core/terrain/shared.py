from types import ModuleType
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model
from scipy.ndimage import zoom

from loco_mujoco.core.terrain.base import hfield_surface_height
from loco_mujoco.core.terrain.static import StaticTerrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class SharedTerrainState:
    """
    Per-env state of the shared terrain.

    Attributes:
        origin_xy: World xy of the centre of the env's current tile (where the env spawns).
        level: Row of the current tile (difficulty level).
        col: Column of the current tile (terrain type).
    """
    origin_xy: Union[np.ndarray, jax.Array]
    level: Union[int, jax.Array]
    col: Union[int, jax.Array]


class SharedTerrain(StaticTerrain):
    """
    One large heightfield shared by all envs, built from a grid of tiles (mjlab/Isaac-Lab style).

    Rows are difficulty levels (row 0 = easiest), columns cycle through the tile types. Every env is assigned a
    tile and spawns on the flat platform at the tile's centre. All platforms are at world height 0, so only the
    xy origin differs between envs. The heightfield never changes after construction, so it lives once in the
    model and is read by every world. This works on MuJoCo (CPU), MJX and MjWarp.

    Tile types (heights in metres, relative to the platform):
        - "flat": flat ground.
        - "rough": random uniform noise, amplitude grows with difficulty.
        - "slope_up" / "slope_down": pyramid slope rising/falling away from the platform.
        - "stairs_up" / "stairs_down": pyramid stairs rising/falling away from the platform.
        - "wave": sinusoidal waves, amplitude grows with difficulty.

    Curriculum (optional): on reset an env moves one level up if its episode lasted at least
    ``promote_frac * horizon`` steps, and one level down if it lasted less than ``demote_frac * horizon``.
    Envs that are promoted beyond the last level are sent to a random level.
    """

    TILE_TYPES = ("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "wave")

    def __init__(self, env: Any,
                 num_rows: int = 10,
                 num_cols: int = 14,
                 tile_size: float = 8.0,
                 horizontal_scale: float = 0.1,
                 border_width: float = 10.0,
                 platform_size: float = 2.0,
                 tile_types: Sequence[str] = ("flat", "rough", "slope_up", "slope_down",
                                              "stairs_up", "stairs_down", "wave"),
                 max_rough_amplitude: float = 0.08,
                 max_slope: float = 0.4,
                 min_step_height: float = 0.03,
                 max_step_height: float = 0.18,
                 step_width: float = 0.35,
                 max_wave_amplitude: float = 0.15,
                 wave_length: float = 2.0,
                 curriculum: bool = True,
                 init_max_level: int = 0,
                 promote_frac: float = 0.8,
                 demote_frac: float = 0.4,
                 seed: int = 0,
                 **kwargs: Any):
        """
        Args:
            env: The environment instance.
            num_rows: Number of difficulty levels.
            num_cols: Number of tile columns; column ``c`` uses ``tile_types[c % len(tile_types)]``.
            tile_size: Edge length of a square tile in metres.
            horizontal_scale: Metres per heightfield cell.
            border_width: Width of the flat border around the tile grid in metres.
            platform_size: Edge length of the flat spawn platform at each tile centre in metres.
            tile_types: Tile types to cycle through over the columns.
            max_rough_amplitude: Noise amplitude of "rough" at the hardest level.
            max_slope: Slope (rise/run) of "slope_*" at the hardest level.
            min_step_height: Step height of "stairs_*" at the easiest level.
            max_step_height: Step height of "stairs_*" at the hardest level.
            step_width: Tread depth of "stairs_*".
            max_wave_amplitude: Amplitude of "wave" at the hardest level.
            wave_length: Wavelength of "wave".
            curriculum: Whether to move envs between levels based on episode length.
            init_max_level: Envs start on a random level in ``[0, init_max_level]``.
            promote_frac: Fraction of the horizon an episode must last to move up a level.
            demote_frac: Episodes shorter than this fraction of the horizon move down a level.
            seed: Seed for the terrain generation.
        """
        super().__init__(env, **kwargs)
        for t in tile_types:
            assert t in self.TILE_TYPES, f"Unknown tile type '{t}'. Available: {self.TILE_TYPES}."
        assert 0 <= init_max_level < num_rows, "init_max_level must be in [0, num_rows)."

        self.num_rows, self.num_cols = num_rows, num_cols
        self.tile_size = tile_size
        self.hs = horizontal_scale
        self.border_width = border_width
        self.platform_size = platform_size
        self.tile_types = tuple(tile_types)
        self.max_rough_amplitude = max_rough_amplitude
        self.max_slope = max_slope
        self.min_step_height, self.max_step_height = min_step_height, max_step_height
        self.step_width = step_width
        self.max_wave_amplitude = max_wave_amplitude
        self.wave_length = wave_length
        self.curriculum = curriculum
        self.init_max_level = init_max_level
        self.promote_frac, self.demote_frac = promote_frac, demote_frac
        self._rng = np.random.default_rng(seed)

        self.horizon = env.info.horizon

        # build the full heightmap (metres, platform level = 0)
        self.tile_cells = int(round(tile_size / horizontal_scale))
        self.border_cells = int(round(border_width / horizontal_scale))
        self.heights = self._build_heightmap()                      # (nrow=y, ncol=x)
        self.nrow, self.ncol = self.heights.shape
        self.half_x = (self.ncol - 1) * self.hs / 2
        self.half_y = (self.nrow - 1) * self.hs / 2

        # tile centres in world coordinates, shape (num_rows, num_cols, 2). Tile row = level, along world x;
        # tile column = type, along world y.
        x0 = -self.half_x + self.border_cells * self.hs
        y0 = -self.half_y + self.border_cells * self.hs
        lvl = np.arange(num_rows)
        col = np.arange(num_cols)
        cx = x0 + (lvl * self.tile_cells + (self.tile_cells - 1) / 2) * self.hs
        cy = y0 + (col * self.tile_cells + (self.tile_cells - 1) / 2) * self.hs
        self.tile_centers = np.stack(np.meshgrid(cx, cy, indexing="ij"), axis=-1)
        self._tile_centers_jnp = jnp.asarray(self.tile_centers)
        self._heights_jnp = jnp.asarray(self.heights)

    # ------------------------------------------------------------------ terrain generation

    def _build_heightmap(self) -> np.ndarray:
        n, b = self.tile_cells, self.border_cells
        H = np.zeros((self.num_cols * n + 2 * b + 1, self.num_rows * n + 2 * b + 1))  # (y, x)
        for lvl in range(self.num_rows):
            difficulty = lvl / max(self.num_rows - 1, 1)
            for col in range(self.num_cols):
                kind = self.tile_types[col % len(self.tile_types)]
                tile = self._make_tile(kind, difficulty)
                H[b + col * n: b + (col + 1) * n, b + lvl * n: b + (lvl + 1) * n] = tile
        return H

    def _make_tile(self, kind: str, d: float) -> np.ndarray:
        n, hs = self.tile_cells, self.hs
        c = (np.arange(n) + 0.5) * hs - n * hs / 2           # cell centres relative to the tile centre
        yy, xx = np.meshgrid(c, c, indexing="ij")
        cheb = np.maximum(np.abs(xx), np.abs(yy))             # square-pyramid distance from the centre
        out = np.maximum(cheb - self.platform_size / 2, 0.0)  # distance from the platform edge

        if kind == "flat":
            h = np.zeros((n, n))
        elif kind == "rough":
            amp = d * self.max_rough_amplitude
            coarse = self._rng.uniform(-amp, amp, size=(max(n // 4, 2), max(n // 4, 2)))
            h = zoom(coarse, n / coarse.shape[0], order=1)[:n, :n]
        elif kind in ("slope_up", "slope_down"):
            h = d * self.max_slope * out
            h = h if kind == "slope_up" else -h
        elif kind in ("stairs_up", "stairs_down"):
            step_h = self.min_step_height + d * (self.max_step_height - self.min_step_height)
            h = step_h * np.ceil(out / self.step_width)
            h = h if kind == "stairs_up" else -h
        elif kind == "wave":
            amp = d * self.max_wave_amplitude
            k = 2 * np.pi / self.wave_length
            h = amp * (np.sin(k * xx) + np.sin(k * yy)) / 2
        else:
            raise ValueError(kind)

        # flatten the spawn platform to height 0
        h = np.where(cheb <= self.platform_size / 2, 0.0, h)
        return h

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        hmin, hmax = float(self.heights.min()), float(self.heights.max())
        z_range = max(hmax - hmin, 1e-3)
        normalized = (self.heights - hmin) / z_range
        spec.add_hfield(name="shared_terrain",
                        size=[self.half_x, self.half_y, z_range, 1.0],
                        nrow=self.nrow, ncol=self.ncol,
                        userdata=normalized.astype(np.float32).ravel().tolist())
        for i, field in enumerate(spec.hfields):
            if field.name == "shared_terrain":
                self.hfield_id = i
                break

        for g in spec.geoms:
            if g.name == "floor":
                spec.delete(g)
                break

        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="shared_terrain",
                                group=2, pos=(0, 0, hmin), material="MatPlane", rgba=(0.8, 0.9, 0.8, 1))
        return spec

    # ------------------------------------------------------------------ per-env state

    def init_state(self, env: Any, key: Any, model: Union[MjModel, Model], data: Union[MjData, Data],
                   backend: ModuleType, traj=None) -> SharedTerrainState:
        assert_backend_is_supported(backend)
        if backend == jnp:
            k1, k2 = jax.random.split(key)
            level = jax.random.randint(k1, (), 0, self.init_max_level + 1)
            col = jax.random.randint(k2, (), 0, self.num_cols)
            origin = self._tile_centers_jnp[level, col]
        else:
            level = int(np.random.randint(0, self.init_max_level + 1))
            col = int(np.random.randint(0, self.num_cols))
            origin = self.tile_centers[level, col].copy()
        return SharedTerrainState(origin_xy=origin, level=level, col=col)

    def reset(self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], carry: Any,
              backend: ModuleType, traj=None) -> Tuple[Union[MjData, Data], Any]:
        """Picks the env's next tile. The robot is moved onto it by ``shift_to_env_origin`` after the
        initial-state handler has set qpos."""
        assert_backend_is_supported(backend)
        state = carry.terrain_state
        # cur_step_in_episode still holds the length of the episode that just ended (it is 1 on the first reset)
        ep_len = carry.cur_step_in_episode
        ended = ep_len > 1

        if backend == jnp:
            ended = jnp.asarray(ended)
            key, k1, k2 = jax.random.split(carry.key, 3)
            level = state.level
            if self.curriculum:
                up = ended & (ep_len >= self.promote_frac * self.horizon)
                down = ended & (ep_len < self.demote_frac * self.horizon)
                level = level + up.astype(jnp.int32) - down.astype(jnp.int32)
                level = jnp.where(level >= self.num_rows, jax.random.randint(k1, (), 0, self.num_rows), level)
                level = jnp.clip(level, 0, self.num_rows - 1)
            else:
                level = jax.random.randint(k1, (), 0, self.num_rows)
            col = jax.random.randint(k2, (), 0, self.num_cols)
            origin = self._tile_centers_jnp[level, col]
            carry = carry.replace(key=key)
        else:
            level = int(state.level)
            if self.curriculum:
                if ended and ep_len >= self.promote_frac * self.horizon:
                    level += 1
                elif ended and ep_len < self.demote_frac * self.horizon:
                    level -= 1
                if level >= self.num_rows:
                    level = int(np.random.randint(0, self.num_rows))
                level = int(np.clip(level, 0, self.num_rows - 1))
            else:
                level = int(np.random.randint(0, self.num_rows))
            col = int(np.random.randint(0, self.num_cols))
            origin = self.tile_centers[level, col].copy()

        carry = carry.replace(terrain_state=SharedTerrainState(origin_xy=origin, level=level, col=col))
        return data, carry

    def get_env_origin_xy(self, carry: Any, backend: ModuleType) -> Union[np.ndarray, jnp.ndarray]:
        return carry.terrain_state.origin_xy

    def shift_to_env_origin(self, env: Any, data: Union[MjData, Data], carry: Any,
                            backend: ModuleType) -> Union[MjData, Data]:
        """Moves all free joints by the env's origin (called after the initial state has been set)."""
        xy_ids = env.free_jnt_qpos_id[:, :2]                       # (n_free, 2)
        origin = carry.terrain_state.origin_xy
        if backend == jnp:
            return data.replace(qpos=data.qpos.at[xy_ids].add(origin))
        data.qpos[xy_ids] += origin
        return data

    @property
    def has_env_origins(self) -> bool:
        return True

    # ------------------------------------------------------------------ queries

    def get_height_at(self, xy: Union[np.ndarray, jnp.ndarray], backend: ModuleType, env: Any = None,
                      carry: Any = None) -> Union[np.ndarray, jnp.ndarray]:
        """Height of the heightfield surface at world points ``xy`` of shape (..., 2)."""
        H = self._heights_jnp if backend == jnp else self.heights
        fx = (xy[..., 0] + self.half_x) / self.hs
        fy = (xy[..., 1] + self.half_y) / self.hs
        return hfield_surface_height(H, fx, fy, self.nrow, self.ncol, backend)
