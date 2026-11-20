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


# ---------------------------------------------------------------------------
# Download path. The HuggingFace download is replaced by a local file, so the
# branches around it (missing variables file, cache write, retargeting fallback,
# kinematic extension of an incomplete clip) run without network access.
# ---------------------------------------------------------------------------
import jax.numpy as jnp
import mujoco
from huggingface_hub.utils import EntryNotFoundError

from loco_mujoco import RLFactory
from loco_mujoco.core.trajectory import TrajectoryData, TrajectoryInfo, TrajectoryModel
from loco_mujoco.datasets.humanoids.LAFAN1 import load as lafan1_load

_REAL_ENV = "UnitreeH1"


def _serve(monkeypatch, tmp_path, traj, missing_for=()):
    """Fake hub that serves ``traj`` and reports the envs in ``missing_for`` as absent."""
    served = tmp_path / "hub.npz"
    traj.save(str(served))
    requests = []

    def fake_download(repo_id, filename, repo_type):
        requests.append(filename)
        if any(f"/{env}/" in filename for env in missing_for):
            raise EntryNotFoundError(f"{filename} not found")
        return str(served)

    monkeypatch.setattr(lafan1_load, "hf_hub_download", fake_download)
    return requests


def test_download_without_variables_file(tmp_path, monkeypatch, standing_trajectory):
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(tmp_path / "does_not_exist.yaml"))
    requests = _serve(monkeypatch, tmp_path, standing_trajectory)

    traj = load_lafan1_trajectory("MyTestEnv", "walk1")

    assert requests == ["Lafan1/mocap/MyTestEnv/walk1.npz"]
    assert traj.data.qpos.shape == standing_trajectory.data.qpos.shape


def test_download_without_cache_path_in_variables_file(tmp_path, monkeypatch, standing_trajectory):
    import yaml
    var_file = tmp_path / "vars.yaml"
    var_file.write_text(yaml.dump({"SOME_OTHER_KEY": "x"}))
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(var_file))
    requests = _serve(monkeypatch, tmp_path, standing_trajectory)

    load_lafan1_trajectory("MyTestEnv", "walk1")
    assert len(requests) == 1


def test_download_is_written_to_the_cache(lafan1_cache, tmp_path, monkeypatch, standing_trajectory):
    requests = _serve(monkeypatch, tmp_path, standing_trajectory)

    load_lafan1_trajectory("MyTestEnv", "walk1")
    assert os.path.isfile(os.path.join(lafan1_cache, "MyTestEnv", "walk1.npz"))

    load_lafan1_trajectory("MyTestEnv", "walk1")
    assert len(requests) == 1   # second call is served from the cache


def test_missing_dataset_is_retargeted_from_the_source_env(tmp_path, monkeypatch, standing_trajectory):
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(tmp_path / "does_not_exist.yaml"))
    requests = _serve(monkeypatch, tmp_path, standing_trajectory, missing_for=("MyTestEnv",))
    calls = []

    def fake_retarget(source_env, traj_source, target_env):
        calls.append((source_env, target_env))
        return traj_source

    monkeypatch.setattr(lafan1_load, "retarget_traj_from_robot_to_robot", fake_retarget)

    traj = load_lafan1_trajectory("MyTestEnv", "walk1", source_env="MySourceEnv")

    assert requests == ["Lafan1/mocap/MyTestEnv/walk1.npz", "Lafan1/mocap/MySourceEnv/walk1.npz"]
    assert calls == [("MySourceEnv", "MyTestEnv")]
    assert isinstance(traj, Trajectory)


def test_incomplete_clip_is_extended_with_forward_kinematics(tmp_path, monkeypatch, capsys):
    # a variables file without a cache path: the robot configuration loader reads it too
    var_file = tmp_path / "vars.yaml"
    var_file.write_text("SOME_OTHER_KEY: x\n")
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(var_file))

    # a clip that only holds joint positions and velocities, as shipped on the hub
    env, _ = RLFactory.make(_REAL_ENV, headless=True)
    model = env._model
    n_steps = 20
    qpos = jnp.tile(jnp.array(model.qpos0), (n_steps, 1))
    info = TrajectoryInfo(
        joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
        model=TrajectoryModel(njnt=model.njnt, jnt_type=jnp.array(model.jnt_type)),
        frequency=1.0 / env.dt)
    data = TrajectoryData(qpos=qpos, qvel=jnp.zeros((n_steps, model.nv)), split_points=jnp.array([0, n_steps]))
    incomplete = Trajectory(info=info, data=data)
    assert not incomplete.data.is_complete
    _serve(monkeypatch, tmp_path, incomplete)

    capsys.readouterr()   # drop the summary printed by the environment created above
    traj = load_lafan1_trajectory(_REAL_ENV, "walk1")
    # the helper environment built for the forward kinematics does not print a second summary
    assert "Observation Summary" not in capsys.readouterr().out

    # the clip is resampled to the control frequency of the robot's retargeting configuration
    n_out = traj.data.qpos.shape[0]
    assert traj.data.is_complete
    assert n_out > 1
    assert abs(n_out / traj.info.frequency - n_steps * env.dt) <= 2.0 / traj.info.frequency
    assert traj.data.qpos.shape == (n_out, model.nq)
    assert traj.data.xpos.shape == (n_out, model.nbody, 3)
    assert traj.data.site_xpos.shape[0] == n_out
    assert np.allclose(traj.data.qpos, qpos[:n_out], atol=1e-6)


# ---------------------------------------------------------------------------
# Dataset groups. The group lists used to carry the ".csv" suffix of the original
# files, so that a group was looked up as "<name>.csv.npz" and never found.
# ---------------------------------------------------------------------------

def test_lafan1_group_names_have_no_file_suffix():
    from loco_mujoco.datasets.humanoids.LAFAN1 import (
        LAFAN1_LOCOMOTION_DATASETS, LAFAN1_DANCE_DATASETS, LAFAN1_ALL_DATASETS)

    for group in (LAFAN1_LOCOMOTION_DATASETS, LAFAN1_DANCE_DATASETS, LAFAN1_ALL_DATASETS):
        assert len(group) > 0 and len(set(group)) == len(group)
        assert all("." not in name for name in group)
    assert set(LAFAN1_LOCOMOTION_DATASETS) <= set(LAFAN1_ALL_DATASETS)
    assert set(LAFAN1_DANCE_DATASETS) <= set(LAFAN1_ALL_DATASETS)


def test_load_lafan1_strips_csv_suffix(lafan1_cache, standing_trajectory):
    # names with the suffix of the original LAFAN1 files keep working
    _stash(lafan1_cache, "MyTestEnv", "dance1_subject1", standing_trajectory)
    traj = load_lafan1_trajectory("MyTestEnv", "dance1_subject1.csv")
    assert isinstance(traj, Trajectory)


def test_lafan1_dataset_group_loads_from_cache(lafan1_cache, standing_trajectory):
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
    from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_DANCE_DATASETS

    env = DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal", reward_type="NoReward",
                           horizon=1000, gamma=0.99)
    for name in LAFAN1_DANCE_DATASETS:
        _stash(lafan1_cache, type(env).__name__, name, standing_trajectory)

    traj = ImitationFactory.get_lafan1_traj(env, LAFAN1DatasetConf(dataset_group="LAFAN1_DANCE_DATASETS"))
    assert isinstance(traj, Trajectory)
    assert len(traj.data.split_points) - 1 == len(LAFAN1_DANCE_DATASETS)
