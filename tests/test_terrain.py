"""Tests of the terrain interface.

Every built-in terrain is checked against MuJoCo's own collision surface (ray casts), on MuJoCo (CPU) and MJX, and
for the robot's placement on reset. HeightmapTerrain additionally gets its box contact, the per-env box pool and the
reset grounding checked; SharedTerrain its env origins. Everything runs on the offline test humanoid. MuJoCo Warp
checks run only where a GPU and mujoco_warp are available.
"""
import copy
import importlib.util
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.core.observations import HeightScan
from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.terrain.base import hfield_surface_height
from loco_mujoco.core.trajectory import Trajectory

from conftest import _synthetic_traj_from_env
from test_conf import DummyHumamoidEnv

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

# the test height map: 12 m x 8 m at 0.1 m, placed far from the world origin
ORIGIN, CELL, NX, NY = np.array([40.0, -15.0]), 0.1, 121, 81
TREAD, RISER = 0.3, 0.15


def _stairs_map() -> np.ndarray:
    """Two staircases along x (y in 1-3 m and 5-7 m) and a round platform (around (9.5, 4) m), on flat ground."""
    x = np.arange(NX) * CELL
    y = np.arange(NY) * CELL
    X, Y = np.meshgrid(x, y)                                   # (NY, NX): row = y, column = x
    h = np.zeros((NY, NX))
    steps = np.clip(np.floor((X - 3.0) / TREAD + 1e-9) + 1, 0, 6)
    h[(Y >= 1.0) & (Y <= 3.0)] = (steps * RISER)[(Y >= 1.0) & (Y <= 3.0)]
    steps = np.clip(np.floor((X - 2.0) / 0.4 + 1e-9) + 1, 0, 8)
    h[(Y >= 5.0) & (Y <= 7.0)] = (steps * 0.12)[(Y >= 5.0) & (Y <= 7.0)]
    h[(X - 9.5) ** 2 + (Y - 4.0) ** 2 <= 0.8 ** 2] = 0.2
    return h


# root paths of the test clips, in map coordinates (metres from ORIGIN)
CLIP_PATHS = [((1.0, 2.0), (7.0, 2.0)), ((1.0, 6.0), (7.0, 6.0)), ((8.0, 4.0), (10.5, 4.0))]


@pytest.fixture(scope="module")
def map_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("heightmap")
    np.savez(d / "map.npz", h=_stairs_map(), cell=CELL, origin=ORIGIN)
    return str(d / "map.npz"), str(d / "cache")


def terrain_conf(name: str, map_file=None, **extra):
    """(terrain_type, terrain_params) of a test case name."""
    if name.startswith("HeightmapTerrain"):
        file, cache = map_file
        params = dict(file=file, cache_dir=cache, tile_size=3.0, contact=name.split("-")[1])
        return "HeightmapTerrain", {**params, **extra}
    if name == "SharedTerrain":
        return name, dict(num_rows=2, num_cols=7, tile_size=4.0, border_width=2.0, init_max_level=1, **extra)
    return name, dict(extra)


def _robot_xml() -> str:
    """The test humanoid without contact margins: MJX has no heightfield-capsule collision with a margin."""
    src = Path(DummyHumamoidEnv.get_default_xml_file_path())
    path = Path(tempfile.mkdtemp()) / src.name
    path.write_text(src.read_text().replace('margin="0.001"', 'margin="0"'))
    return str(path)


ROBOT_XML = _robot_xml()


def make_env(name, map_file=None, enable_mjx=False, terrain_extra=None, **kwargs):
    ttype, params = terrain_conf(name, map_file, **(terrain_extra or {}))
    return DummyHumamoidEnv(spec=ROBOT_XML, enable_mjx=enable_mjx, terrain_type=ttype, terrain_params=params,
                            **DEFAULTS, **kwargs)


def clip_trajectory(env, n_steps: int = 60) -> Trajectory:
    """The test clips: the test humanoid's standing frame moved along CLIP_PATHS (world coordinates). The root height
    stays the standing height of flat ground, so on the stairs the robot starts inside a step and the terrain has
    to lift it (``ground_robot``)."""
    base = _synthetic_traj_from_env(env, n_steps)
    trajs = []
    for (x0, y0), (x1, y1) in CLIP_PATHS:
        s = np.linspace(0.0, 1.0, n_steps)[:, None]
        xy = ORIGIN + (1 - s) * np.array([x0, y0]) + s * np.array([x1, y1])        # (n_steps, 2)
        d = base.data
        shift = np.zeros((n_steps, 3))
        shift[:, :2] = xy - np.asarray(d.qpos[:, :2])
        qpos = np.asarray(d.qpos).copy()
        qpos[:, :3] += shift
        data = d.replace(qpos=jnp.asarray(qpos),
                         xpos=d.xpos.at[:, 1:].add(shift[:, None]),
                         subtree_com=d.subtree_com.at[:, 1:].add(shift[:, None]),
                         site_xpos=d.site_xpos + shift[:, None])
        trajs.append(base.replace(data=data))
    return Trajectory.concatenate(trajs)


def ray_heights(model, data, xy, z0=20.0):
    """Height of the static (world-body) collision surface under each point, by ray casts from z0 downwards."""
    out = np.zeros(len(xy))
    for k, (x, y) in enumerate(xy):
        gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        assert dist >= 0 and model.geom_bodyid[gid[0]] == 0, f"ray at {(x, y)} hit {gid[0]}"
        out[k] = z0 - dist
    return out


def robot_out_of_the_way(env):
    """A copy of the env's model and data for ray casts against the terrain's collision surface: the robot lifted
    above the ray start, and world geoms without collisions (e.g. HeightmapTerrain's height-query tiles under its
    boxes) made transparent, which rays skip."""
    m = copy.deepcopy(env._model)
    off = (m.geom_bodyid == 0) & (m.geom_contype == 0) & (m.geom_conaffinity == 0)
    m.geom_rgba[off, 3] = 0.0
    m.geom_matid[off] = -1
    d = mujoco.MjData(m)
    d.qpos[:] = env._data.qpos
    d.qpos[2] = 100.0
    mujoco.mj_forward(m, d)
    return m, d


def sample_points(env, name, n, rng):
    """Random points on the terrain, away from box edges (where a ray could hit either of two boxes)."""
    t = env._terrain
    if name.startswith("HeightmapTerrain"):
        ij = np.stack([rng.integers(0, NX, n), rng.integers(0, NY, n)], -1)
        return ORIGIN + np.clip(ij + rng.uniform(-0.4, 0.4, (n, 2)), 0, [NX - 1, NY - 1]) * CELL
    if name == "SharedTerrain":
        return rng.uniform(-1, 1, (n, 2)) * np.array([t.half_x, t.half_y]) * 0.99
    if name == "RoughTerrain":
        return rng.uniform(-3.9, 3.9, (n, 2))
    return rng.uniform(-5, 5, (n, 2))


ALL = ["StaticTerrain", "RoughTerrain", "SharedTerrain", "HeightmapTerrain-hfield", "HeightmapTerrain-boxes"]


def test_every_registered_terrain_is_tested():
    assert set(Terrain.list_registered()) == {n.split("-")[0] for n in ALL}


# ------------------------------------------------------------------------------------------------ height queries

def test_hfield_surface_height_is_mujocos_surface():
    """The helper interpolates on MuJoCo's triangles: exact against ray casts, unlike bilinear interpolation."""
    rng = np.random.default_rng(0)
    nr, nc, sx, sy, sz, pos = 5, 7, 3.0, 2.0, 0.5, np.array([0.3, -0.2, 0.1])
    xml = f"""<mujoco><asset><hfield name="h" nrow="{nr}" ncol="{nc}" size="{sx} {sy} {sz} 0.1"/></asset>
    <worldbody><geom type="hfield" hfield="h" pos="{pos[0]} {pos[1]} {pos[2]}"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    H = rng.uniform(0, 1, (nr, nc))
    m.hfield_data[:] = H.ravel()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xy = pos[:2] + rng.uniform(-1, 1, (500, 2)) * np.array([sx, sy]) * 0.999
    fx = (xy[:, 0] - pos[0] + sx) / (2 * sx) * (nc - 1)
    fy = (xy[:, 1] - pos[1] + sy) / (2 * sy) * (nr - 1)
    for backend in (np, jnp):
        h = pos[2] + sz * np.asarray(hfield_surface_height(backend.asarray(H), backend.asarray(fx),
                                                           backend.asarray(fy), nr, nc, backend))
        np.testing.assert_allclose(h, ray_heights(m, d, xy), atol=1e-5)
    near = hfield_surface_height(H, fx, fy, nr, nc, np, nearest=True)
    np.testing.assert_array_equal(near, H[np.round(fy).astype(int), np.round(fx).astype(int)])


@pytest.mark.parametrize("name", ALL)
def test_heights_match_the_collision_surface(name, map_file):
    env = make_env(name, map_file)
    env.reset()
    env.step(np.zeros(env.info.action_space.shape))         # per-env model changes are applied in update()
    m, d = robot_out_of_the_way(env)
    xy = sample_points(env, name, 300, np.random.default_rng(1))
    h = env._terrain.get_height_at(xy, np, env=env, carry=env._additional_carry)
    assert h.shape == (300,)
    np.testing.assert_allclose(h, ray_heights(m, d, xy), atol=1e-4)
    env.stop()


@pytest.mark.parametrize("name", ALL)
def test_mjx_heights_and_steps(name, map_file):
    """Jitted JAX heights equal the NumPy ones, and vmapped resets and steps run."""
    env = make_env(name, map_file, enable_mjx=True)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    state = jax.jit(jax.vmap(env.mjx_reset))(keys)
    state = jax.jit(jax.vmap(env.mjx_step))(state, jnp.zeros((3, env.info.action_space.shape[0])))
    assert np.isfinite(np.asarray(state.data.qpos)).all()

    xy = sample_points(env, name, 200, np.random.default_rng(2))
    t = env._terrain
    carry = jax.tree.map(lambda x: x[0], state.additional_carry)
    h_jax = jax.jit(lambda c: t.get_height_at(jnp.asarray(xy), jnp, env=env, carry=c))(carry)
    h_np = t.get_height_at(xy, np, env=env, carry=jax.tree.map(np.asarray, carry))
    np.testing.assert_allclose(np.asarray(h_jax), h_np, atol=1e-5)


def test_rough_terrain_needs_the_carry():
    env = make_env("RoughTerrain")
    with pytest.raises(AssertionError, match="carry"):
        env._terrain.get_height_at(np.zeros((1, 2)), np)


def test_dynamic_terrains_are_refused_on_mjwarp():
    assert make_env("RoughTerrain")._terrain.is_dynamic
    with pytest.raises(ValueError, match="per-env heightfield"):
        make_env("RoughTerrain", enable_mjx=True, use_mjwarp=True)


# where the robot stands for the height scan: on rough ground, on the first staircase
SCAN_XY = {"RoughTerrain": np.array([2.0, 1.5]), "HeightmapTerrain-boxes": ORIGIN + np.array([4.0, 2.0])}


@pytest.mark.parametrize("name", ["StaticTerrain", "RoughTerrain", "SharedTerrain", "HeightmapTerrain-boxes"])
def test_height_scan_reads_the_terrain(name, map_file):
    env = make_env(name, map_file)
    env.reset()
    env.step(np.zeros(env.info.action_space.shape))         # per-env model changes are applied in update()
    if name in SCAN_XY:
        env._data.qpos[:2] = SCAN_XY[name]
        mujoco.mj_forward(env._model, env._data)
    scan = HeightScan("scan", size=(1.0, 0.6), resolution=0.2, clip=(-5.0, 5.0))
    scan._init_from_mj(env, env._model, env._data, 0)
    obs, _ = scan.get_obs_and_update_state(env, env._model, env._data, env._additional_carry, np)
    q = env._data.qpos
    w, x, y, z = q[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = q[:2] + scan._grid @ rot.T
    m, d = robot_out_of_the_way(env)
    np.testing.assert_allclose(obs, q[2] - ray_heights(m, d, pts), atol=1e-4)
    env.stop()


# ------------------------------------------------------------------------------------------------ placement

def test_recenter_root_xy_follows_the_terrain(map_file):
    assert make_env("StaticTerrain")._recenter_root_xy
    assert make_env("SharedTerrain")._recenter_root_xy
    assert not make_env("HeightmapTerrain-boxes", map_file)._recenter_root_xy
    assert not make_env("StaticTerrain", recenter_root_xy=False)._recenter_root_xy
    with pytest.raises(ValueError, match="recenter_root_xy=False"):
        make_env("HeightmapTerrain-boxes", map_file, recenter_root_xy=True)
    with pytest.raises(ValueError, match="recenter_root_xy=True"):
        make_env("SharedTerrain", recenter_root_xy=False)


@pytest.mark.parametrize("name", ["StaticTerrain", "SharedTerrain", "HeightmapTerrain-boxes"])
def test_reference_and_robot_share_the_world_frame(name, map_file):
    """On reset the robot starts at the reference root (trajectory xy + offset); the trajectory terminal check and
    the offset agree on every terrain."""
    env = make_env(name, map_file, init_state_type="TrajInitialStateHandler",
                   terminal_state_type="RootPoseTrajTerminalStateHandler",
                   th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    traj = clip_trajectory(env)
    env.process_trajectory(traj)
    for k in range(len(CLIP_PATHS)):
        env.th.fixed_start_conf = (k, 0)
        env.reset()
        carry = env._additional_carry
        ref_xy = np.asarray(traj.data.qpos)[np.asarray(traj.data.split_points)[k], :2]
        offset = np.asarray(env.get_traj_offset_xy(carry, np))
        np.testing.assert_allclose(env._data.qpos[:2], ref_xy + offset, atol=1e-6)
        if name == "HeightmapTerrain-boxes":
            np.testing.assert_allclose(offset, 0.0)            # clips start where they are on the map
        elif name == "StaticTerrain":
            np.testing.assert_allclose(env._data.qpos[:2], 0.0, atol=1e-6)
        else:
            np.testing.assert_allclose(env._data.qpos[:2], carry.terrain_state.origin_xy, atol=1e-6)
        excess_pos = env._terminal_state_handler._violations(env, env._data, carry, np, env._traj)[0][0]
        assert excess_pos + env._terminal_state_handler.max_root_pos_deviation < 1e-6
    env.stop()


def test_shared_terrain_spreads_envs_over_flat_platforms():
    env = make_env("SharedTerrain", enable_mjx=True)
    t = env._terrain
    offsets = np.linspace(-t.platform_size / 2 + 0.1, t.platform_size / 2 - 0.1, 5)
    pts = t.tile_centers[:, :, None, None, :] + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), -1)
    assert np.abs(t.get_height_at(pts.reshape(-1, 2), np)).max() < 1e-9
    state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.PRNGKey(0), 32))
    origin = np.asarray(state.additional_carry.terrain_state.origin_xy)
    np.testing.assert_allclose(np.asarray(state.data.qpos[:, :2]), origin + env._init_model.qpos0[:2], atol=1e-5)
    assert len(np.unique(origin, axis=0)) > 5


def test_heightmap_lifts_the_robot_out_of_the_stairs(map_file):
    """A clip whose root height ignores the stairs is lifted on reset, so no terrain-colliding geom starts inside a
    step."""
    env = make_env("HeightmapTerrain-boxes", map_file, init_state_type="TrajInitialStateHandler",
                   th_params=dict(random_start=False, fixed_start_conf=(0, 40)))
    env.process_trajectory(clip_trajectory(env))
    t = env._terrain
    env.reset()                                               # frame 40 of clip 0 is on the stairs
    ids, pts, rad, _ = t._ground_setup(env)
    d = env._data
    p = d.geom_xpos[ids][:, None, :] + np.einsum("gij,gkj->gki", d.geom_xmat[ids].reshape(-1, 3, 3), pts)
    gap = p[..., 2] - rad - t.get_height_at(p[..., :2], np)
    assert t.get_height_at(d.qpos[None, :2], np)[0] > 0.3
    assert gap.min() >= t.ground_clearance - 1e-6
    env.stop()


# ------------------------------------------------------------------------------------------------ HeightmapTerrain

def test_heightmap_tiles_and_boxes_cover_the_map(map_file):
    t = make_env("HeightmapTerrain-boxes", map_file)._terrain
    assert (t.nty, t.ntx) == (3, 4)
    B = t.boxes()
    h = _stairs_map()
    # every cell centre lies in exactly one box (or on the ground), at the box's height
    cx = ORIGIN[0] + np.arange(NX) * CELL
    cy = ORIGIN[1] + np.arange(NY) * CELL
    X, Y = np.meshgrid(cx, cy)
    inside = ((B[:, 0] < X[..., None]) & (X[..., None] < B[:, 1]) & (B[:, 2] < Y[..., None]) & (Y[..., None] < B[:, 3]))
    assert inside.sum(-1).max() == 1
    top = np.where(inside.any(-1), (inside * B[:, 5]).sum(-1), h.min())
    np.testing.assert_allclose(top, h, atol=t.box_quant / 2 + 1e-9)


def pool_env(map_file, tmp_path, reach=0.5, enable_mjx=False, n_traj=None, **kwargs):
    """A pool env with the test clips loaded (its pool file built from them)."""
    base = make_env("HeightmapTerrain-boxes", map_file)
    traj = clip_trajectory(base)
    t = base._terrain
    path = str(tmp_path / f"pool_{reach}.npz")
    t.write_pool_file(path, traj if n_traj is None else Trajectory.concatenate([traj] * n_traj), reach=reach)
    kwargs.setdefault("init_state_type", "TrajInitialStateHandler")
    env = make_env("HeightmapTerrain-pool", map_file, enable_mjx=enable_mjx, terrain_extra=dict(pool_file=path),
                   **kwargs)
    return env, traj, base


def test_pool_file_lists_the_boxes_near_each_clip(map_file, tmp_path):
    base = make_env("HeightmapTerrain-boxes", map_file)
    traj = clip_trajectory(base)
    B = base._terrain.boxes()
    idx = base._terrain.write_pool_file(str(tmp_path / "p.npz"), traj, reach=0.5)
    assert idx.shape[0] == len(CLIP_PATHS)
    sp = np.asarray(traj.data.split_points)
    for k in range(len(CLIP_PATHS)):
        p = np.asarray(traj.data.qpos[sp[k]:sp[k + 1], :2])
        dx = np.maximum(np.maximum(B[:, 0] - p[:, None, 0], p[:, None, 0] - B[:, 1]), 0)
        dy = np.maximum(np.maximum(B[:, 2] - p[:, None, 1], p[:, None, 1] - B[:, 3]), 0)
        near = np.flatnonzero((np.hypot(dx, dy) <= 0.5).any(0))
        np.testing.assert_array_equal(np.sort(idx[k][idx[k] >= 0]), near)
        assert len(near) > 0


def test_pool_gives_the_full_map_contacts_near_the_clip(map_file, tmp_path):
    env, traj, base = pool_env(map_file, tmp_path, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    env.process_trajectory(traj)
    rng = np.random.default_rng(3)
    for k, ((x0, y0), (x1, y1)) in enumerate(CLIP_PATHS):
        env.th.fixed_start_conf = (k, 0)
        env.reset()
        env.step(np.zeros(env.info.action_space.shape))      # the slots are written in update()
        m, d = robot_out_of_the_way(env)
        # points near the root path, away from box edges
        s = rng.uniform(0, 1, 300)[:, None]
        xy = ORIGIN + (1 - s) * np.array([x0, y0]) + s * np.array([x1, y1]) + rng.uniform(-0.3, 0.3, (300, 2))
        ij = np.clip(np.round((xy - ORIGIN) / CELL), 0, [NX - 1, NY - 1])
        xy = ORIGIN + (ij + rng.uniform(-0.4, 0.4, (300, 2))) * CELL      # within 0.5 m (the reach) of the path
        np.testing.assert_allclose(ray_heights(m, d, xy), base._terrain.get_height_at(xy, np), atol=1e-4)
    env.stop()


def test_pool_mjx_writes_each_envs_own_boxes(map_file, tmp_path):
    env, traj, _ = pool_env(map_file, tmp_path, enable_mjx=True)
    env.process_trajectory(traj)
    t = env._terrain
    state = jax.jit(jax.vmap(env.mjx_reset))(jax.random.split(jax.random.PRNGKey(0), 16))
    traj_no = np.asarray(state.additional_carry.traj_state.traj_no)
    assert len(np.unique(traj_no)) > 1
    sys = env.sys
    model, data, _ = jax.jit(jax.vmap(lambda d, c: t.update(env, sys, d, c, jnp, env._traj)))(
        state.data, state.additional_carry)
    g = slice(t._pool_geoms, t._pool_geoms + t.pool_slots)
    np.testing.assert_allclose(np.asarray(model.geom_pos[:, g]), t.pool_ctr[traj_no], atol=1e-6)
    np.testing.assert_allclose(np.asarray(model.geom_size[:, g]), t.pool_half[traj_no], atol=1e-6)
    np.testing.assert_allclose(np.asarray(data.geom_xpos[:, g]), t.pool_ctr[traj_no], atol=1e-6)
    state = jax.jit(jax.vmap(env.mjx_step))(state, jnp.zeros((16, env.info.action_space.shape[0])))
    assert np.isfinite(np.asarray(state.data.qpos)).all()


def test_pool_errors(map_file, tmp_path):
    env, traj, _ = pool_env(map_file, tmp_path, n_traj=2,
                            th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    env.process_trajectory(traj)
    with pytest.raises(ValueError, match="has boxes for 6 trajectories, the dataset has 3"):
        env.reset()
        env.step(np.zeros(env.info.action_space.shape))

    env, _, _ = pool_env(map_file, tmp_path, init_state_type="DefaultInitialStateHandler")
    with pytest.raises(ValueError, match="no trajectory state"):
        env.reset()
        env.step(np.zeros(env.info.action_space.shape))

    other = tmp_path / "other.npz"
    np.savez(other, idx=np.zeros((3, 2), np.int32), n_boxes=1)
    with pytest.raises(AssertionError, match="indexes 1 boxes"):
        make_env("HeightmapTerrain-pool", map_file, terrain_extra=dict(pool_file=str(other)))


def test_pool_steps_like_the_full_box_map(map_file, tmp_path):
    """MuJoCo (float64): from a pose pushed 1 cm into the stairs, 20 steps with a pool that holds every box the
    falling robot can reach give the state of 20 steps with every box of the map. The starts switch between the
    clips, so the slots (and MuJoCo's bounding volume tree of the world body) must follow the trajectory."""
    th = dict(random_start=False, fixed_start_conf=(0, 0))
    boxes = make_env("HeightmapTerrain-boxes", map_file, init_state_type="TrajInitialStateHandler", th_params=th)
    traj = clip_trajectory(boxes)
    boxes.process_trajectory(traj)
    pool, _, _ = pool_env(map_file, tmp_path, reach=3.0, th_params=th)
    pool.process_trajectory(traj)
    assert pool._terrain.pool_slots < len(boxes._terrain.boxes())
    act = np.zeros(boxes.info.action_space.shape)
    for start in [(0, 30), (0, 45), (1, 40), (2, 30), (0, 10)]:     # on the stairs and next to the round platform
        out = []
        for env in (boxes, pool):
            env.th.fixed_start_conf = start
            env.reset()
            env._data.qpos[2] -= 0.01
            env.step(act)
            ncon = env._data.ncon
            for _ in range(19):
                env.step(act)
            out.append((env._data.qpos.copy(), env._data.qvel.copy(), ncon))
        (q_boxes, v_boxes, n_boxes), (q_pool, v_pool, n_pool) = out
        assert n_boxes > 0 and n_pool == n_boxes
        np.testing.assert_allclose(q_pool, q_boxes, atol=1e-6)
        np.testing.assert_allclose(v_pool, v_boxes, atol=1e-4)
    boxes.stop()
    pool.stop()


# ------------------------------------------------------------------------------------------------ MuJoCo Warp

def _has_mjwarp() -> bool:
    return jax.default_backend() == "gpu" and importlib.util.find_spec("mujoco_warp") is not None


@pytest.mark.skipif(not _has_mjwarp(), reason="needs a GPU and mujoco_warp")
def test_pool_on_mjwarp_matches_the_full_box_map(map_file, tmp_path):
    """MuJoCo Warp, the same resets in 64 worlds, once with every box in every world and once with the pool: the
    terrain pushes every robot back up from 1 cm inside it, and five steps give the same state up to float32 noise
    (the boxes come in another order). A pool that gave a world the wrong boxes would let its robot fall through or
    stand on another step."""
    kw = dict(enable_mjx=True, use_mjwarp=True, naconmax=64 * 16, init_state_type="TrajInitialStateHandler")
    boxes = make_env("HeightmapTerrain-boxes", map_file, **kw)
    traj = clip_trajectory(boxes)
    boxes.process_trajectory(traj)
    pool, _, _ = pool_env(map_file, tmp_path, reach=3.0, **kw)
    pool.process_trajectory(traj)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    act = jnp.zeros((64, boxes.info.action_space.shape[0]))
    out = []
    for env in (boxes, pool):
        state = jax.jit(jax.vmap(env.mjx_reset))(keys)
        assert len(np.unique(np.asarray(state.additional_carry.traj_state.traj_no))) == 3   # every clip runs
        state = state.replace(data=state.data.replace(qpos=state.data.qpos.at[:, 2].add(-0.01)))
        step = jax.jit(jax.vmap(env.mjx_step))
        state = step(state, act)
        assert (np.asarray(state.data.qvel[:, 2]) > 0).all()
        for _ in range(4):
            state = step(state, act)
        out.append(np.asarray(state.data.qpos))
    q_boxes, q_pool = out
    assert np.isfinite(q_boxes).all()
    np.testing.assert_allclose(q_pool, q_boxes, atol=1e-4)
