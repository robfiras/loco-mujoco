"""Offline tests for ``ImitationFactory.get_default_traj``.

The real method downloads the default dataset from the HuggingFace Hub. Here the
download is replaced by a routine that serves a locally saved synthetic trajectory,
so the cache handling, the multi-task concatenation and the retargeting fallback
around it run without network access.
"""
import jax
import pytest
import yaml
from huggingface_hub.utils import EntryNotFoundError

import loco_mujoco
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.task_factories import ImitationFactory, imitation_factory
from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf

from conftest import _synthetic_traj_from_env
from test_conf import DummyHumamoidEnv

jax.config.update("jax_platform_name", "cpu")

# conftest replaces get_default_traj for every test; keep the real one, which is what we test
_get_default_traj = ImitationFactory.__dict__["get_default_traj"].__func__

_N_STEPS = 50


@pytest.fixture
def env():
    return DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal", reward_type="NoReward",
                            horizon=1000, gamma=0.99, n_envs=1)


@pytest.fixture
def hub(env, tmp_path, monkeypatch):
    """Fake HuggingFace Hub: serves one synthetic trajectory and records the requests."""
    served = tmp_path / "hub" / "traj.npz"
    served.parent.mkdir()
    _synthetic_traj_from_env(env, n_steps=_N_STEPS).save(str(served))
    requests = []

    def fake_download(repo_id, filename, repo_type):
        requests.append(filename)
        return str(served)

    monkeypatch.setattr(imitation_factory, "hf_hub_download", fake_download)
    return requests


def _use_variables_file(monkeypatch, tmp_path, content=None):
    path = tmp_path / "LOCOMUJOCO_VARIABLES.yaml"
    if content is not None:
        path.write_text(yaml.dump(content))
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", path)
    return path


def test_downloads_when_no_variables_file_exists(env, hub, tmp_path, monkeypatch):
    _use_variables_file(monkeypatch, tmp_path)   # file is not created

    traj = _get_default_traj(env, DefaultDatasetConf(task="walk"))

    assert hub == ["DefaultDatasets/mocap/DummyHumamoidEnv/walk.npz"]
    assert isinstance(traj, Trajectory)
    assert traj.data.qpos.shape[1] == env._model.nq


def test_several_tasks_are_concatenated(env, hub, tmp_path, monkeypatch):
    _use_variables_file(monkeypatch, tmp_path, content={"SOME_OTHER_KEY": "x"})   # no cache path set

    single = _get_default_traj(env, DefaultDatasetConf(task="walk"))
    both = _get_default_traj(env, DefaultDatasetConf(task=["walk", "run"]))

    assert hub[-2:] == ["DefaultDatasets/mocap/DummyHumamoidEnv/walk.npz",
                        "DefaultDatasets/mocap/DummyHumamoidEnv/run.npz"]
    assert both.data.qpos.shape[0] == 2 * single.data.qpos.shape[0]
    assert len(both.data.split_points) == 3


def test_download_is_cached_and_reused(env, hub, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _use_variables_file(monkeypatch, tmp_path, content={"LOCOMUJOCO_CONVERTED_DEFAULT_PATH": str(cache)})

    first = _get_default_traj(env, DefaultDatasetConf(task="walk"))
    assert len(hub) == 1
    assert (cache / "mocap" / "DummyHumamoidEnv" / "walk.npz").is_file()

    second = _get_default_traj(env, DefaultDatasetConf(task="walk"))
    assert len(hub) == 1   # served from the cache, no second download
    assert second.data.qpos.shape == first.data.qpos.shape


def test_falls_back_to_retargeting_when_dataset_is_not_on_the_hub(env, tmp_path, monkeypatch):
    _use_variables_file(monkeypatch, tmp_path)
    calls = []

    def missing(repo_id, filename, repo_type):
        raise EntryNotFoundError(f"{filename} not found")

    def fake_retarget(env_name, task, dataset_type, source_env):
        calls.append((env_name, task, dataset_type, source_env))
        return _synthetic_traj_from_env(env, n_steps=_N_STEPS)

    monkeypatch.setattr(imitation_factory, "hf_hub_download", missing)
    monkeypatch.setattr(ImitationFactory, "_retarget_default_on_the_fly", staticmethod(fake_retarget))

    traj = _get_default_traj(env, DefaultDatasetConf(task="walk"))

    assert calls == [("DummyHumamoidEnv", "walk", "mocap", imitation_factory._DEFAULT_RETARGET_SOURCE_ENV)]
    assert traj.data.qpos.shape[1] == env._model.nq
