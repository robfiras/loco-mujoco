"""Offline coverage for the LAFAN1 loader cache-hit path.

``load_lafan1_trajectory`` normally downloads from HuggingFace, but when
``LOCOMUJOCO_CONVERTED_LAFAN1_PATH`` is set and a converted ``.npz`` already
exists under ``<cache>/<env_name>/<dataset>.npz`` it loads straight from disk and
never touches the network. We drive exactly that branch (including the ``.npz``
suffix strip, the ``Mjx`` env-name strip, and the multi-dataset concatenate
branch) by pre-populating a temp cache with the synthetic ``standing_trajectory``.
"""
import os

import numpy as np
import jax
import pytest

import loco_mujoco
from test_conf import *  # noqa: F401,F403  (standing_trajectory fixture)
from loco_mujoco.datasets.humanoids.LAFAN1.load import load_lafan1_trajectory
from loco_mujoco.core.trajectory import Trajectory

jax.config.update('jax_platform_name', 'cpu')


@pytest.fixture
def lafan1_cache(tmp_path, monkeypatch):
    """A variables file pointing LOCOMUJOCO_CONVERTED_LAFAN1_PATH at a temp dir."""
    import yaml
    cache_dir = tmp_path / "lafan1_cache"
    cache_dir.mkdir()
    var_file = tmp_path / "vars.yaml"
    with open(var_file, "w") as f:
        yaml.dump({"LOCOMUJOCO_CONVERTED_LAFAN1_PATH": str(cache_dir)}, f)
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(var_file))
    return cache_dir


def _stash(cache_dir, env_name, d_name, traj):
    dst = os.path.join(cache_dir, env_name, f"{d_name}.npz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    traj.save(dst)
    return dst


def test_load_lafan1_from_cache_single(lafan1_cache, standing_trajectory):
    _stash(lafan1_cache, "MyTestEnv", "dance1", standing_trajectory)

    # ".npz" suffix on the requested name is stripped before the cache lookup
    traj = load_lafan1_trajectory("MyTestEnv", "dance1.npz")

    assert isinstance(traj, Trajectory)
    assert np.all(np.isfinite(np.asarray(traj.data.qpos)))


def test_load_lafan1_from_cache_mjx_name(lafan1_cache, standing_trajectory):
    # the "Mjx" infix in the env name is stripped -> file stored under the base name
    _stash(lafan1_cache, "MyTestEnv", "walk1", standing_trajectory)

    traj = load_lafan1_trajectory("MyTestEnvMjx", ["walk1"])
    assert isinstance(traj, Trajectory)


def test_load_lafan1_from_cache_concatenate(lafan1_cache, standing_trajectory):
    # two cached datasets -> exercises the concatenate branch
    _stash(lafan1_cache, "MyTestEnv", "walk1", standing_trajectory)
    _stash(lafan1_cache, "MyTestEnv", "walk2", standing_trajectory)

    traj = load_lafan1_trajectory("MyTestEnv", ["walk1", "walk2"])
    assert isinstance(traj, Trajectory)
    # concatenation doubles the number of samples
    single = load_lafan1_trajectory("MyTestEnv", "walk1")
    assert traj.data.qpos.shape[0] == 2 * single.data.qpos.shape[0]
