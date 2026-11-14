import jax
import mujoco
import numpy as np
import pytest

from loco_mujoco.task_factories import RLFactory

TERRAIN = dict(terrain_type="SharedTerrain",
               terrain_params=dict(num_rows=3, num_cols=7, tile_size=6.0, init_max_level=2))


@pytest.fixture(scope="module")
def cpu_env():
    env, _ = RLFactory.make("UnitreeH1", headless=True, **TERRAIN)
    yield env
    env.stop()


def test_heightfield_matches_heightmap(cpu_env):
    t = cpu_env._terrain
    m = cpu_env._model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rng = np.random.default_rng(0)
    for r, c in zip(rng.integers(0, t.nrow, 100), rng.integers(0, t.ncol, 100)):
        x, y = -t.half_x + c * t.hs, -t.half_y + r * t.hs
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, np.array([x, y, 10.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert geomid[0] == floor
        assert abs((10.0 - dist) - t.heights[r, c]) < 1e-4
        assert abs(t.get_height_at(np.array([x, y]), np) - t.heights[r, c]) < 1e-6


def test_spawn_platforms_are_flat_at_zero(cpu_env):
    t = cpu_env._terrain
    offsets = np.linspace(-t.platform_size / 2 + 0.1, t.platform_size / 2 - 0.1, 5)
    pts = t.tile_centers[:, :, None, None, :] + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), -1)
    assert np.abs(t.get_height_at(pts.reshape(-1, 2), np)).max() < 1e-9


def test_cpu_reset_places_robot_at_origin(cpu_env):
    for _ in range(3):
        cpu_env.reset()
        origin = cpu_env._additional_carry.terrain_state.origin_xy
        xy_ids = cpu_env.free_jnt_qpos_id[0, :2]
        np.testing.assert_allclose(cpu_env._data.qpos[xy_ids], origin + cpu_env._init_model.qpos0[xy_ids],
                                   atol=1e-6)


def test_mjx_reset_spreads_envs_over_tiles():
    env, _ = RLFactory.make("MjxUnitreeH1", headless=True, **TERRAIN)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    state = jax.jit(jax.vmap(env.mjx_reset))(keys)
    origin = np.asarray(state.additional_carry.terrain_state.origin_xy)
    xy_ids = env.free_jnt_qpos_id[0, :2]
    np.testing.assert_allclose(np.asarray(state.data.qpos[:, xy_ids]), origin + env._init_model.qpos0[xy_ids],
                               atol=1e-5)
    assert len(np.unique(origin, axis=0)) > 5
    state = jax.jit(jax.vmap(env.mjx_step))(state, np.zeros((32, env.info.action_space.shape[0])))
    assert np.isfinite(np.asarray(state.data.qpos)).all()
